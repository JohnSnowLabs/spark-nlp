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
  * ==Greedy, not [[com.johnsnowlabs.ml.ai.util.Generation.Generate]]==
  *
  * The reference implementation is strictly greedy (`num_beams=1`, `do_sample=False`,
  * `repetition_penalty=1.1`, `bad_words_ids=[[unk]]`), and the shared `Generate` trait has no
  * hook for a KV cache. Extending it would also drag in machinery we would then have to work
  * around: `RepetitionPenaltyLogitProcessor` computes `inputIds.head.distinct` once and applies
  * row 0's history to every row of the batch, and `NoRepeatNgramsLogitProcessor` allocates a
  * vocab-sized structure per row per step. So this is a purpose-built loop.
  *
  * ==Why the cache is mandatory==
  *
  * Cross-attention K/V over the encoder's 784 states costs ~16.4 GFLOP and, uncached, is
  * recomputed every step. For a 1000-token table that is ~261 TFLOP versus ~0.5 TFLOP cached —
  * about 500x.
  *
  * ==The 40/20 asymmetry==
  *
  * `decoder_model.onnx` (prefill) returns '''40''' `present.*` tensors: 10 layers x {encoder,
  * decoder} x {key, value}. `decoder_with_past_model.onnx` consumes all 40 as `past_key_values.*`
  * but returns only the '''20''' `present.*.decoder.*`, because the encoder K/V is computed once
  * from `encoder_hidden_states` and never changes. So the encoder half is held fixed for the
  * whole generation and only the decoder half rolls. (`Whisper.initDecoderOnnx` assigns both
  * halves the same key list; that is a bug, not a pattern to copy.)
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

    /** `present.3.decoder.key` -> `past_key_values.3.decoder.key` */
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

  /** Read only the final position's logits out of a `[batch, seqLen, vocab]` tensor.
    *
    * Deliberately reads through the `FloatBuffer` rather than `getFloatArray`: at batch 16 with a
    * 10-token prompt the full prefill logits are ~47 MB, and we need 16 rows of 73,921 floats.
    */
  private def readLastPositionLogits(
      tensor: OnnxTensor,
      batchSize: Int,
      seqLen: Int,
      into: Array[Array[Float]]): Unit = {
    Dolphin.requireVocabDim(tensor.getInfo.getShape, batchSize, seqLen, vocabSize)
    val buffer = tensor.getFloatBuffer
    var row = 0
    while (row < batchSize) {
      // Widen to java.nio.Buffer before position(): FloatBuffer.position(int) gained a covariant
      // return type in Java 9, so calling it directly bakes in a signature that throws
      // NoSuchMethodError on a Java 8 runtime.
      (buffer: java.nio.Buffer).position(row * seqLen * vocabSize + (seqLen - 1) * vocabSize)
      buffer.get(into(row))
      row += 1
    }
  }

  /** HuggingFace `RepetitionPenaltyLogitsProcessor`, in place and '''per row'''. Divides when the
    * logit is positive and multiplies when negative, so the penalty always discourages.
    */
  private def applyRepetitionPenalty(
      logits: Array[Float],
      seen: mutable.BitSet,
      penalty: Double): Unit =
    if (penalty != 1.0d) {
      // Compute in Float, not Double. PyTorch treats the Python scalar in `score * self.penalty`
      // as a weak type and evaluates in float32; doing it in float64 and rounding afterwards
      // differs in the last bit, which over a long generation is enough to flip one near-tie
      // argmax. Caught on a ~1500-token table where it changed a single character.
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

      // One buffer for the whole generation. At batch 4 with a 73,921 vocabulary this is ~1.2 MB;
      // allocating it per row per step instead churned gigabytes over a long table.
      val logits = Array.ofDim[Float](batchSize, vocabSize)
      readLastPositionLogits(
        prefillResult.get(Sig.logits).get().asInstanceOf[OnnxTensor],
        batchSize,
        promptLength,
        logits)

      // Session output names do not change between steps; scanning and filtering them every
      // iteration was pure overhead.
      val pastPresentKeys = Sig.presentKeys(pastSession)

      // The encoder half of the cache is fixed for the whole generation, so seed the input map once
      // and only replace what actually changes.
      val decoderInputs = new java.util.HashMap[String, OnnxTensor](64)
      encoderPast.foreach { case (k, v) => decoderInputs.put(k, v) }

      // Seed the penalty history with the prompt, matching HuggingFace.
      val seen = Array.fill(batchSize)(mutable.BitSet(promptIds: _*))
      val finished = Array.fill(batchSize)(false)

      var step = 0
      while (step < budget && !finished.forall(identity)) {
        // `forced_eos_token_id` in generation_config.json: HuggingFace forces EOS at
        // `max_length - 1`, so a generation that hits the cap ends with EOS, one content token
        // short of the cap.
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
        if (finished.forall(identity) || step >= budget) {
          // nothing left to feed the decoder
        } else {
          val inputIds = OnnxTensor.createTensor(pastEnv, next.map(Array(_)))
          decoderInputs.put(Sig.decoderInputIds, inputIds)
          decoderPast.foreach { case (k, v) => decoderInputs.put(k, v) }
          val stepResult =
            try pastSession.run(decoderInputs)
            finally inputIds.close()

          // Only now is the previous step's cache superseded and safe to release.
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

  /** mBART's learned position table (`max_position_embeddings`). The window covers prompt and
    * output together, which is what upstream's `max_length=4096` means; the table cannot be
    * extended without retraining.
    */
  val MaxPositions: Int = 4096

  /** The decoder emits `[batch, seqLen, vocab]` logits and `readLastPositionLogits` strides the
    * buffer using the `vocab_size` from `config.json`. A disagreement is not recoverable: the
    * reads land mid-row and the model decodes plausible-looking garbage, so fail loudly at the
    * first forward pass instead.
    */
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
