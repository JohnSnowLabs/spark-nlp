/*
 * Copyright 2017-2026 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.ai

import ai.onnxruntime.{OnnxTensor, OrtSession}
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxSession
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWrappers
import com.johnsnowlabs.ml.onnx.TensorResources.implicits._
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.transform.DonutImageUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.DolphinTokenizer

import java.awt.image.BufferedImage
import scala.collection.JavaConverters._
import scala.collection.mutable

/** Inference for ByteDance Dolphin: a Swin (donut-swin) vision encoder feeding an mBart decoder,
  * driven by a task prompt supplied as the decoder prefix.
  *
  * The reference implementation is strictly greedy (`num_beams=1`, `do_sample=False`,
  * `repetition_penalty=1.1`, `bad_words_ids=[[unk]]`), and the shared `Generate` trait has no
  * hook for a KV cache. Extending it would also drag in machinery unnescary machineary:
  * `RepetitionPenaltyLogitProcessor` computes `inputIds.head.distinct` once and applies row 0's
  * history to every row of the batch, and `NoRepeatNgramsLogitProcessor` allocates a vocab-sized
  * structure per row per step. So this is a purpose-built loop.
  */
private[johnsnowlabs] class Dolphin(
    val onnxWrappers: EncoderDecoderWrappers,
    val tokenizer: DolphinTokenizer,
    val preprocessor: Preprocessor,
    val generationConfig: GenerationConfig)
    extends Serializable {

  private val GenerationConfig(_, paddingTokenId, eosTokenId, vocabSize, _, _, _) =
    generationConfig

  /** `bad_words_ids=[[unk]]` in the reference: the model must never emit `<unk>`. */
  private val unkTokenId: Int = tokenizer.specialTokens.unk.id

  @transient private lazy val onnxSessionOptions: Map[String, String] =
    new OnnxSession().getSessionOptions

  private object Sig {
    val encoderInput = "pixel_values"
    val encoderOutput = "last_hidden_state"
    val decoderInputIds = "input_ids"
    val decoderEncoderState = "encoder_hidden_states"
    val logits = "logits"

    def presentKeys(session: OrtSession): Array[String] =
      session.getOutputNames.asScala.filter(_.startsWith("present.")).toArray

    def toPast(key: String): String = key.replace("present", "past_key_values")

    def isEncoderState(key: String): Boolean = key.contains(".encoder.")
  }

  // ---------------------------------------------------------------- preprocessing

  /** @return NCHW float tensor, `[batch][3][size][size]`, RGB */
  def preprocessImages(images: Array[BufferedImage]): Array[Array[Array[Array[Float]]]] =
    images.map { image =>
      DonutImageUtils.pixelValues(
        image = image,
        size = preprocessor.size,
        resample = preprocessor.resample,
        mean = preprocessor.image_mean,
        std = preprocessor.image_std,
        doNormalize = preprocessor.do_normalize,
        doRescale = preprocessor.do_rescale,
        rescaleFactor = preprocessor.rescale_factor)
    }

  // ---------------------------------------------------------------- helpers

  private def argmax(scores: Array[Float]): Int = {
    var best = 0
    var i = 1
    while (i < scores.length) {
      if (scores(i) > scores(best)) best = i
      i += 1
    }
    best
  }

  /** Read only the final position's logits out of a `[batch, seqLen, vocab]` tensor. */
  private def readLastPositionLogits(
      tensor: OnnxTensor,
      batchSize: Int,
      seqLen: Int,
      into: Array[Array[Float]]): Unit = {
    Dolphin.requireVocabDim(tensor.getInfo.getShape, batchSize, seqLen, vocabSize)
    val buffer = tensor.getFloatBuffer
    var row = 0
    while (row < batchSize) {
      (buffer: java.nio.Buffer).position(row * seqLen * vocabSize + (seqLen - 1) * vocabSize)
      buffer.get(into(row))
      row += 1
    }
  }

  private def applyRepetitionPenalty(
      logits: Array[Float],
      seen: mutable.BitSet,
      penalty: Double): Unit =
    if (penalty != 1.0d) {
      val p = penalty.toFloat
      seen.foreach { id =>
        val v = logits(id)
        logits(id) = if (v < 0) v * p else v / p
      }
    }

  // ---------------------------------------------------------------- generation

  /** Run the two ONNX graphs for one batch of crops that all share the same task prompt.
    *
    * All rows must share a prompt: the exported decoder has '''no attention-mask input''', so
    * ragged prompt lengths in one batch are not expressible. Callers group elements by type,
    * which is what makes the prompts uniform.
    *
    * @return
    *   generated ids per row, prompt prefix already removed
    */
  def generate(
      images: Array[BufferedImage],
      promptIds: Array[Int],
      maxNewTokens: Int,
      repetitionPenalty: Double): Array[Array[Int]] = {

    require(images.nonEmpty, "generate called with no images")

    val batchSize = images.length
    val promptLength = promptIds.length
    val budget = math.min(maxNewTokens, Dolphin.MaxPositions - promptLength)
    require(budget > 0, s"a $promptLength-token prompt leaves no room in the decoder window")

    val (encoderSession, encoderEnv) = onnxWrappers.encoder.getSession(onnxSessionOptions)
    val (decoderSession, decoderEnv) = onnxWrappers.decoder.getSession(onnxSessionOptions)
    val (pastSession, pastEnv) = onnxWrappers.decoderWithPast.getSession(onnxSessionOptions)

    // ---- encoder ------------------------------------------------------------
    val pixelValues = OnnxTensor.createTensor(encoderEnv, preprocessImages(images))
    val encoderResult =
      try encoderSession.run(Map(Sig.encoderInput -> pixelValues).asJava)
      finally pixelValues.close()

    val generated = Array.fill(batchSize)(mutable.ArrayBuffer.empty[Int])
    var prefillResult: OrtSession.Result = null
    var previousStep: OrtSession.Result = null

    try {
      val encoderStates =
        encoderResult.get(Sig.encoderOutput).get().asInstanceOf[OnnxTensor]

      // ---- prefill ----------------------------------------------------------
      val promptBatch = Array.fill(batchSize)(promptIds.map(_.toLong))
      val promptTensor = OnnxTensor.createTensor(decoderEnv, promptBatch)
      prefillResult =
        try
          decoderSession.run(
            Map(
              Sig.decoderInputIds -> promptTensor,
              Sig.decoderEncoderState -> encoderStates).asJava)
        finally promptTensor.close()

      val prefillKeys = Sig.presentKeys(decoderSession)
      val prefillStates = prefillResult.getOnnxTensors(prefillKeys)

      // The encoder half never changes; only the decoder half rolls forward.
      val encoderPast: Map[String, OnnxTensor] =
        prefillStates.filterKeys(Sig.isEncoderState).map { case (k, v) => Sig.toPast(k) -> v }
      var decoderPast: Map[String, OnnxTensor] =
        prefillStates.filterNot(kv => Sig.isEncoderState(kv._1)).map { case (k, v) =>
          Sig.toPast(k) -> v
        }

      val logits = Array.ofDim[Float](batchSize, vocabSize)
      readLastPositionLogits(
        prefillResult.get(Sig.logits).get().asInstanceOf[OnnxTensor],
        batchSize,
        promptLength,
        logits)

      val pastPresentKeys = Sig.presentKeys(pastSession)

      val decoderInputs = new java.util.HashMap[String, OnnxTensor](64)
      encoderPast.foreach { case (k, v) => decoderInputs.put(k, v) }

      val seen = Array.fill(batchSize)(mutable.BitSet(promptIds: _*))
      val finished = Array.fill(batchSize)(false)

      var step = 0
      while (step < budget && !finished.forall(identity)) {
        val forceEos = step == budget - 1
        val next = new Array[Long](batchSize)
        var row = 0
        while (row < batchSize) {
          if (finished(row)) next(row) = paddingTokenId.toLong
          else {
            val token =
              if (forceEos) eosTokenId
              else {
                val rowLogits = logits(row)
                rowLogits(unkTokenId) = Float.NegativeInfinity
                applyRepetitionPenalty(rowLogits, seen(row), repetitionPenalty)
                argmax(rowLogits)
              }
            if (token == eosTokenId) finished(row) = true
            else {
              generated(row) += token
              seen(row) += token
            }
            next(row) = token.toLong
          }
          row += 1
        }
        step += 1
        if (finished.forall(identity) || step >= budget) {} else {
          val inputIds = OnnxTensor.createTensor(pastEnv, next.map(Array(_)))
          decoderInputs.put(Sig.decoderInputIds, inputIds)
          decoderPast.foreach { case (k, v) => decoderInputs.put(k, v) }
          val stepResult =
            try pastSession.run(decoderInputs)
            finally inputIds.close()

          if (previousStep != null) previousStep.close()
          previousStep = stepResult

          decoderPast = stepResult
            .getOnnxTensors(pastPresentKeys)
            .map { case (k, v) => Sig.toPast(k) -> v }
          readLastPositionLogits(
            stepResult.get(Sig.logits).get().asInstanceOf[OnnxTensor],
            batchSize,
            1,
            logits)
        }
      }
    } finally {
      if (previousStep != null) previousStep.close()
      if (prefillResult != null) prefillResult.close()
      encoderResult.close()
    }

    generated.map(_.toArray)
  }

  /** Generate and decode in one go. */
  def parse(
      images: Array[BufferedImage],
      prompt: String,
      maxNewTokens: Int,
      repetitionPenalty: Double): Array[String] = {
    val promptIds =
      DolphinTokenizer.GoldenPromptIds.getOrElse(prompt, tokenizer.encodePrompt(prompt))
    generate(images, promptIds, maxNewTokens, repetitionPenalty)
      .map(ids => tokenizer.decodeTokens(ids.filter(_ != paddingTokenId)).trim)
  }
}

private[johnsnowlabs] object Dolphin {

  val MaxPositions: Int = 4096

  def requireVocabDim(shape: Array[Long], batchSize: Int, seqLen: Int, vocabSize: Int): Unit = {
    require(
      shape.length == 3,
      s"expected [batch, seq, vocab] logits but got a rank-${shape.length} tensor")
    require(
      shape(0) == batchSize.toLong && shape(1) == seqLen.toLong,
      s"expected logits [$batchSize, $seqLen, *] but got [${shape.mkString(", ")}]")
    require(
      shape(2) == vocabSize.toLong,
      s"the decoder emits ${shape(2)} logits per position but the generation config declares " +
        s"vocab_size $vocabSize; the ONNX export and config.json disagree")
  }
}
