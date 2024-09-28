/*
 * Copyright 2017-2022 John Snow Labs
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

import ai.onnxruntime.OnnxTensor
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common.{Sentence, SentenceSplit}
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.Gpt2Tokenizer
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.tensorflow.Session

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.math.exp

private[johnsnowlabs] class GPT2(
    val tensorflow: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val bpeTokenizer: Gpt2Tokenizer,
    configProtoBytes: Option[Array[Byte]] = None)
    extends Serializable {

  // keys representing the input and output tensors of the GPT2 model
  private val inputIdsKey = "serving1_serving1_input_ids:0"
  private val attentionMaskKey = "serving1_serving1_attention_mask:0"
  private val outputLogitsKey = "StatefulPartitionedCall:0"
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions
  private val paddingTokenId = 50256
  private val eosTokenId = 50256
  val detectedEngine: String =
    if (tensorflow.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else ONNX.name

  private def sessionWarmup(): Unit = {
    val dummyInput = Array.fill(128)(0) ++ Array(eosTokenId)
    tag(
      Seq(dummyInput),
      minOutputLength = 1,
      maxOutputLength = 5,
      doSample = false,
      temperature = 0f,
      topK = 0,
      topP = 0f,
      repetitionPenalty = 0f,
      noRepeatNgramSize = 0,
      randomSeed = Option(0),
      ignoreTokenIds = Array(paddingTokenId))
  }

  sessionWarmup()

  def predict(
      sentences: Seq[Annotation],
      batchSize: Int,
      minOutputLength: Int,
      maxOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      task: String,
      randomSeed: Option[Int] = None,
      ignoreTokenIds: Array[Int] = Array()): Seq[Annotation] = {

    val batchDecoder = sentences.grouped(batchSize).toArray.flatMap { batch =>
      val batchSP = encode(batch, task)

      val spIds = tag(
        batchSP,
        minOutputLength,
        maxOutputLength,
        doSample,
        temperature,
        topK,
        topP,
        repetitionPenalty,
        noRepeatNgramSize,
        randomSeed,
        ignoreTokenIds)
      decode(spIds)
    }

    var sentBegin, nextSentEnd = 0

    batchDecoder.zip(sentences).map { case (content, sent) =>
      nextSentEnd += content.length - 1
      val annots = new Annotation(
        annotatorType = AnnotatorType.DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = sent.metadata)
      sentBegin += nextSentEnd + 1
      annots
    }

  }

  def tag(
      batch: Seq[Array[Int]],
      minOutputLength: Int,
      maxOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Int],
      ignoreTokenIds: Array[Int] = Array()): Array[Array[Int]] = {

    val numReturn_sequences = 1
    // from config
    val vocab_size = 50257

    var effectiveBatch_size = 1

    // set effective batch size and effective batch multiplier according to do_sample
    if (doSample) {
      effectiveBatch_size = batch.length * numReturn_sequences
    } else {
      effectiveBatch_size = batch.length
    }

    val maxSentenceLength = batch.map(_.length).max

    val paddedBatch = batch.map { tokenIds =>
      val diff = maxSentenceLength - tokenIds.length
      Array.fill[Int](diff)(this.paddingTokenId) ++ tokenIds.take(maxSentenceLength)
    }

    generateNoBeamSearch(
      paddedBatch,
      maxOutputLength,
      minOutputLength,
      doSample,
      temperature,
      topK,
      topP,
      repetitionPenalty,
      noRepeatNgramSize,
      effectiveBatch_size,
      vocab_size,
      randomSeed,
      ignoreTokenIds)
  }

  def generateNoBeamSearch(
      inputIds: Seq[Array[Int]],
      maxOutputLength: Int,
      minOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      batch_size: Int,
      vocab_size: Int,
      randomSeed: Option[Int],
      ignoreTokenIds: Array[Int] = Array()): Array[Array[Int]] = {

    /** Generate sequences for each example without beam search (numBeams == 1). All returned
      * sequence are generated independently.
      */
    var decoderInputs = inputIds.toArray

    var curLen = decoderInputs(0).length

    var stopDecoder = false

    // length of generated sentences / unfinished sentences
    var unfinishedSents = List.fill(decoderInputs.length)(1)
    var sentLengths = List.fill(decoderInputs.length)(maxOutputLength)
    var decoderOutputs: Array[Array[Array[Float]]] = Array.empty

    while (!stopDecoder) {
      val decoderInputLength = decoderInputs.head.length
      if (detectedEngine == TensorFlow.name) {
        val tensorDecoder = new TensorResources()
        val session = tensorflow.get.getTFSessionWithSignature(
          configProtoBytes = configProtoBytes,
          initAllTables = false)

        val decoderInputBuffers =
          tensorDecoder.createIntBuffer(decoderInputs.length * decoderInputLength)
        val decoderAttentionBuffers =
          tensorDecoder.createIntBuffer(decoderInputs.length * decoderInputLength)

        decoderInputs.zipWithIndex.foreach { case (pieceIds, idx) =>
          val offset = idx * decoderInputLength
          decoderInputBuffers.offset(offset).write(pieceIds)
          val paddingMasks = pieceIds.map(_ => 1)
          decoderAttentionBuffers.offset(offset).write(paddingMasks)
        }

        val inputIdTensors = tensorDecoder.createIntBufferTensor(
          Array(decoderInputs.length.toLong, decoderInputLength),
          decoderInputBuffers)
        val attentionMaskTensors = tensorDecoder.createIntBufferTensor(
          Array(decoderInputs.length.toLong, decoderInputLength),
          decoderAttentionBuffers)
        val runner = session.runner

        // TODO add past to the model and use cache
        runner
          .feed(inputIdsKey, inputIdTensors)
          .feed(attentionMaskKey, attentionMaskTensors)
          .fetch(outputLogitsKey)

        val decoderOuts = runner.run().asScala
        decoderOutputs = TensorResources
          .extractFloats(decoderOuts.head)
          .grouped(vocab_size)
          .toArray
          .grouped(decoderInputLength)
          .toArray

        decoderOuts.foreach(_.close())
        tensorDecoder.clearTensors()
        tensorDecoder.clearSession(decoderOuts)
        inputIdTensors.close()
      } else {
        val (session, env) = onnxWrapper.get.getSession(onnxSessionOptions)

        val decoderInputBuffers = decoderInputs
          .map(tokenIds => tokenIds.map(_.toLong))
        val decoderPaddingBuffers =
          decoderInputBuffers.map(x => x.map(xx => 1L))

        val inputPositionIDsLong: Array[Array[Long]] =
          decoderInputs.map { tokenIds =>
            tokenIds.zipWithIndex.map { case (_, i) =>
              i.toLong
            }
          }

        val decoderPositionIDs: OnnxTensor =
          OnnxTensor.createTensor(env, inputPositionIDsLong)

        val decoderInputTensors = OnnxTensor.createTensor(env, decoderInputBuffers)
        val decoderPaddingMaskTensors = OnnxTensor.createTensor(env, decoderPaddingBuffers)

        val decoderResults = session.run(
          mapAsJavaMap(
            Map(
              "input_ids" -> decoderInputTensors,
              "attention_mask" -> decoderPaddingMaskTensors,
              "position_ids" -> decoderPositionIDs)))

        val decoderOuts = decoderResults
          .get("logits")
          .get()
          .asInstanceOf[OnnxTensor]
        decoderOutputs = decoderOuts.getFloatBuffer
          .array()
          .grouped(vocab_size)
          .toArray
          .grouped(decoderInputLength)
          .toArray

        decoderInputTensors.close()
        decoderPaddingMaskTensors.close()
        decoderPositionIDs.close()
        decoderOuts.close()

      }
      var nextTokenLogits = for (decoderOutput <- decoderOutputs) yield decoderOutput.last

      nextTokenLogits = nextTokenLogits.map(logits => {
        logits.indices
          .map(i => {
            if (ignoreTokenIds.contains(i)) Float.MinValue else logits(i)
          })
          .toArray
      })

      // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
      if (repetitionPenalty != 1.0) {
        nextTokenLogits =
          createNextTokenLogitsPenalties(decoderInputs, nextTokenLogits, repetitionPenalty)
      }

      if (noRepeatNgramSize > 0) {
        // calculate a list of banned tokens to prevent repetitively generating the same ngrams
        // from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        val bannedTokens =
          calcBannedNgramTokens(decoderInputs, batch_size, noRepeatNgramSize, curLen)
        // create bannedTokens boolean mask
        var bannedTokensIndicesMask = Array.empty[IndexedSeq[Boolean]]
        for (bannedTokensSlice <- bannedTokens) {
          bannedTokensIndicesMask = bannedTokensIndicesMask :+
            (for (token <- 0 until vocab_size)
              yield if (bannedTokensSlice.contains(token)) true else false)
        }
        if (!bannedTokensIndicesMask.isEmpty) {
          nextTokenLogits =
            for ((nextTokenLogit, bannedTokensIndexMask) <- nextTokenLogits.zip(
                bannedTokensIndicesMask))
              yield setTensorByIndicesToValue(
                nextTokenLogit,
                bannedTokensIndexMask,
                Float.NegativeInfinity)
        }
      }

      // set eos token prob to zero if minLength is not reached
      if (!eosTokenId.isNaN && curLen < minOutputLength) {
        // create eosTokenId boolean mask
        val isTokenLogit_eosToken =
          for (token <- 0 until vocab_size)
            yield if (token == eosTokenId) true else false

        val eosTokenIndices_mask = Array.fill(batch_size)(isTokenLogit_eosToken)

        nextTokenLogits =
          for ((nextTokenLogit, bannedTokensIndex_mask) <- nextTokenLogits.zip(
              eosTokenIndices_mask))
            yield setTensorByIndicesToValue(
              nextTokenLogit,
              bannedTokensIndex_mask,
              Float.NegativeInfinity)
      }

      var nextToken = Array.ofDim[Int](decoderInputs.length)

      if (doSample) {
        // Temperature (higher temperature => more likely to sample low probability tokens). May not be 0
        if (temperature != 1.0 && temperature > 0)
          nextTokenLogits =
            for (nextTokenLogit <- nextTokenLogits)
              yield nextTokenLogit.map(_ / temperature.toFloat)
        // Top-p/top-k filtering
        nextTokenLogits = topKTopPFiltering(nextTokenLogits, topK, topP)
        // Sample
        nextToken = nextTokenLogits.map(input => categoricalSample(input, randomSeed))
      } else {
        // Greedy decoding

        nextToken = nextTokenLogits.map(input => input.indexOf(input.max))
      }
      var tokensToAdd = Array.ofDim[Int](decoderInputs.length)

      // update generations and finished sentences
      if (!eosTokenId.isNaN)
        // pad finished sentences if eos_token_id exist
        tokensToAdd =
          nextToken.zip(unfinishedSents).map(x => x._1 * x._2 + paddingTokenId * (1 - x._2))
      else
        tokensToAdd = nextToken

      decoderInputs = decoderInputs
        .zip(tokensToAdd)
        .map(x => {
          x._1 ++ Array(x._2)
        })

      curLen += 1

      if (!eosTokenId.isNaN) {
        val eosInSents = tokensToAdd.map(x => if (x == eosTokenId) 1 else 0)
        // if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
        val isSentsUnfinishedAndTokenToAddIsEos =
          unfinishedSents.zip(eosInSents).map(x => x._1 * x._2)

        sentLengths = sentLengths
          .zip(isSentsUnfinishedAndTokenToAddIsEos)
          .map(x => x._1 * (1 - x._2) + curLen * x._2)

        // unfinishedSents is set to zero if eos in sentence
        unfinishedSents =
          unfinishedSents.zip(isSentsUnfinishedAndTokenToAddIsEos).map(x => x._1 - x._2)
      }

      // stop when there is a eos in each sentence, or if we exceed the maximum length
      //      stopDecoder = curLen < maxOutputLength || unfinishedSents.max == 0
      stopDecoder = (!decoderInputs.exists(o => o.last != this.eosTokenId)
        || (decoderInputs.head.length > maxOutputLength))
    }
    decoderInputs
  }

  def createNextTokenLogitsPenalties(
      inputIds: Seq[Array[Int]],
      logits: Array[Array[Float]],
      repetitionPenalty: Double): Array[Array[Float]] = {
    // create logit penalties for already seen inputIds
    val nextTokenLogits = Array.ofDim[Array[Float]](logits.length)

    for (i <- logits.indices) {
      var nextTokenLogit = logits(i)
      val prevInputIds = inputIds.head.distinct
      for ((prevInputId, j) <- prevInputIds.zipWithIndex) {
        var logitPenalty = 1.0
        if (logits(i)(prevInputId.toInt) < 0) {
          logitPenalty = repetitionPenalty
        } else {
          logitPenalty = 1 / repetitionPenalty
        }
        nextTokenLogit = nextTokenLogit.updated(
          prevInputId,
          (logitPenalty * nextTokenLogit(prevInputId)).toFloat)
      }
      nextTokenLogits(i) = nextTokenLogit
    }
    nextTokenLogits
  }

  private def calcBannedNgramTokens(
      prevInputIds: Seq[Array[Int]],
      numHypos: Int,
      noRepeatNgramSize: Int,
      curLen: Int): Array[Array[Int]] = {
    // based on fairseq for noRepeatNgram in beam_search
    if (curLen + 1 < noRepeatNgramSize)
      // return no banned tokens if we haven't generated noRepeatNgram_size tokens yet
      return Array.ofDim[Int](numHypos, 0)
    val generatedNgrams =
      Array.tabulate(numHypos)(_ => mutable.Map.empty[IndexedSeq[Int], List[Int]])
    for (idx <- 0 until numHypos) {
      val genTokens = prevInputIds(idx)
      val generatedNgram = generatedNgrams(idx)
      val ngramArrays = for (e <- 0 until noRepeatNgramSize) yield genTokens.drop(e)
      for (ngramInd <- ngramArrays.last.indices) {
        val ngram = for (e <- ngramArrays) yield e(ngramInd)
        val prevNgramTuple = ngram.dropRight(1)
        generatedNgram(prevNgramTuple) =
          generatedNgram.getOrElse(prevNgramTuple, List.empty[Int]) :+ ngram.last
      }
    }
    (for (hypoIdx <- 0 until numHypos)
      yield getGeneratedNgrams(
        prevInputIds,
        generatedNgrams,
        hypoIdx,
        curLen,
        noRepeatNgramSize)).toArray
  }

  def getGeneratedNgrams(
      prevInputIds: Seq[Array[Int]],
      generatedNgrams: Array[mutable.Map[IndexedSeq[Int], List[Int]]],
      hypoIdx: Int,
      curLen: Int,
      noRepeatNgramSize: Int): Array[Int] = {
    // Before decoding the next token, prevent decoding of ngrams that have already appeared
    val startIdx = curLen + 1 - noRepeatNgramSize
    val ngramIdx = prevInputIds(hypoIdx).slice(startIdx, curLen)
    generatedNgrams(hypoIdx).getOrElse(ngramIdx, List.empty[Int]).toArray
  }

  private def topKTopPFiltering(
      logits: Array[Array[Float]],
      topK: Int,
      topP: Double,
      filterValue: Float = Float.NegativeInfinity,
      minTokensToKeep: Int = 1): Array[Array[Float]] = {

    /** Filter a distribution of logits using top-k and/or nucleus (top-p) filtering * Args:
      * logits: logits distribution shape (batch size, vocabulary size) if topK > 0: keep only top
      * k tokens with highest probability (top-k filtering). if topP < 1.0: keep the top tokens
      * with cumulative probability >= topP (nucleus filtering). Nucleus filtering is described in
      * Holtzman et al. (http://arxiv.org/abs/1904.09751) Make sure we keep at least
      * minTokensToKeep per batch example in the output From:
      * https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
      */
    var logitsUpd = logits
    val logitsShape = Array(logits.length, logits(0).length)

    if (topK > 0) {
      val topKup = topK.max(minTokensToKeep).min(logitsShape.last) // Safety check

      /** Remove all tokens with a probability less than the last token of the top-k */
      val removeLimit = logits(0).sortWith(_ > _).take(topKup).min
      val indicesToRemove =
        for (logit <- logits)
          yield for (elem <- logit) yield if (elem < removeLimit) true else false

      logitsUpd =
        for ((nextTokenLogit, indexToRemove) <- logits.zip(indicesToRemove))
          yield setTensorByIndicesToValue(nextTokenLogit, indexToRemove, Float.NegativeInfinity)
    }
    if (topP < 1.0) {
      val (sortedLogits, sortedIndices) = logits(0).zipWithIndex.sorted.reverse.unzip

      val cumulativeProbs = scanLeft(softmax(sortedLogits))(0.0)(_ + _).drop(1)

      /** Remove tokens with cumulative probability above the threshold (token with 0 are kept) */
      var sortedIndicesToRemove =
        for (prob <- cumulativeProbs)
          yield if (prob > topP) true else false

      if (minTokensToKeep > 1) {

        /** Keep at least minTokensToKeep (set to minTokensToKeep-1 because we add the first one
          * below)
          */
        sortedIndicesToRemove = List.fill(sortedIndicesToRemove.take(minTokensToKeep).length)(
          false) ++ sortedIndicesToRemove.drop(minTokensToKeep)
      }

      /** Shift the indices to the right to keep also the first token above the threshold */
      sortedIndicesToRemove = sortedIndicesToRemove.takeRight(1) ++ sortedIndicesToRemove
        .dropRight(1)
      sortedIndicesToRemove =
        List.fill(sortedIndicesToRemove.take(1).length)(false) ++ sortedIndicesToRemove
          .drop(1)

      /** scatter sorted tensors to original indexing */
      val indicesToRemove = scatterValuesOnBatchIndices(sortedIndicesToRemove, sortedIndices)
      logitsUpd =
        for ((nextTokenLogit, indexToRemove) <- logits.zip(
            IndexedSeq.fill(logits.length)(indicesToRemove)))
          yield setTensorByIndicesToValue(
            nextTokenLogit,
            indexToRemove.toIndexedSeq,
            Float.NegativeInfinity)
    }
    logitsUpd
  }

  private def scanLeft[a, b](xs: Iterable[a])(s: b)(f: (b, a) => b) =
    xs.foldLeft(List(s))((acc, x) => f(acc.head, x) :: acc).reverse

  private def scatterValuesOnBatchIndices(
      values: List[Boolean],
      batchIndices: Array[Int]): List[Boolean] = {
    // scatter values to pair indices
    val (_, initArray) = batchIndices.zip(values).sorted.unzip
    initArray.toList
  }

  private def softmax(values: Array[Float]): Array[Float] = {
    val expElem = values.map(exp(_))
    val total = expElem.sum
    expElem.map(_ / total).map(_.toFloat)
  }

  private def setTensorByIndicesToValue(
      prevInputIds: Array[Float],
      indices: IndexedSeq[Boolean],
      value: Float): Array[Float] = {
    for ((inputId, index) <- prevInputIds.zip(indices)) yield if (index) value else inputId
  }

  private def categoricalSample(dist: Array[Float], randomSeed: Option[Int]): Int = {
    val (distFiltered, indices) =
      dist.zipWithIndex.filter { case (elem, index) => !elem.isInfinite }.sorted.unzip

    if (distFiltered.length == 1)
      return indices(0)

    //    val distMinValue = distFiltered.min
    //    val distRange = distFiltered.max - distMinValue
    //    val normalized = distFiltered.map(i => (i - distMinValue)/distRange)
    val normalized = softmax(distFiltered)

    var randomDouble = 0.0
    if (randomSeed.isDefined)
      randomDouble = new scala.util.Random(randomSeed.get).nextDouble()
    else
      randomDouble = scala.util.Random.nextDouble()

    var accum = 0.0
    for ((itemProb, i) <- normalized.zip(indices)) {
      accum += itemProb
      if (accum >= randomDouble) {
        return i
      }
    }
    if (distFiltered.length == 0) {
      // TODO should distFiltered.length == 0 happen (?)
      //  Can we something better than 0?
      return 0
    }
    indices(0)
  }

  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map(s => bpeTokenizer.decodeTokens(s))
  }

  def encode(sentences: Seq[Annotation], task: String): Seq[Array[Int]] = {
    SentenceSplit
      .unpack(sentences)
      .map(s => {
        val sentWithTask =
          if (task.nonEmpty)
            new Sentence(
              content = task.concat(" ").concat(s.content),
              start = s.start,
              end = s.end + task.length + 1,
              index = s.index,
              metadata = s.metadata)
          else s
        bpeTokenizer.tokenize(sentWithTask).map(bpeTokenizer.encode).flatMap(_.map(_.pieceId))
      })
  }

}
