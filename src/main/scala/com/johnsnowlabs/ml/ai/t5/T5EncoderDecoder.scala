package com.johnsnowlabs.ml.ai.t5

import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.mutable
import scala.math.exp

abstract class T5EncoderDecoder(
                                 val spp: SentencePieceWrapper,
                                 val additionalTokens: Map[Int, String] = Map()
                               ) {

  protected val paddingTokenId = 0
  protected  val eosTokenId = 1
  protected  val pieceSize: Int = spp.getSppModel.getPieceSize
  protected val vocabSize = 32128

  def sessionWarmup(): Unit = {
    val dummyInput = Array(1079, 5682, eosTokenId)

    tag(
      Seq(dummyInput),
      maxNewTokens = 10,
      maxTextLength = 10,
      doSample = false,
      temperature = 0f,
      topK = 0,
      topP = 0f,
      repetitionPenalty = 0f,
      randomSeed = Option(0),
      ignoreTokenIds = Array(paddingTokenId),
      stopAtEos = true,
      noRepeatNgramSize = 0
    )
  }

  protected def decode(sentences: Array[Array[Int]]): Seq[String] = {

    sentences.map { s =>
      val filteredPieceIds = s.filter(x => x <= pieceSize || additionalTokens.contains(x))
      val additionalTokenPositions = filteredPieceIds.zipWithIndex.filter(x => additionalTokens.contains(x._1)).map(_._2)
      val decodedStrings = if (additionalTokenPositions.nonEmpty){
        var offset = 0
        additionalTokenPositions.map(i => {
          val slice = spp.getSppModel.decodeIds(filteredPieceIds.slice(offset, i).map(_.toInt): _*) + additionalTokens(filteredPieceIds(i))
          offset = i + 1
          slice
        }) ++ Array(
          spp.getSppModel.decodeIds(filteredPieceIds.slice(offset, filteredPieceIds.length).map(_.toInt): _*)
        )
      } else {
        Array(spp.getSppModel.decodeIds(filteredPieceIds.map(_.toInt): _*))
      }
      decodedStrings.mkString("")
    }

  }

  protected def encode(sentences: Seq[Annotation], task: String): Seq[Array[Int]] = {
    sentences.map(s => {
      val sentWithTask = if (task.nonEmpty) task.concat(" ").concat(s.result) else s.result
      spp.getSppModel.encodeAsIds(sentWithTask) ++ Array(this.eosTokenId)
    })
  }

  protected def encodeS(sentences: Seq[String], task: String): Seq[Array[Int]] = {
    sentences.map(s => {
      val sentWithTask = if (task.nonEmpty) task.concat(" ").concat(s) else s
      spp.getSppModel.encodeAsIds(sentWithTask) ++ Array(this.eosTokenId)
    })
  }


  protected def getGeneratedNgrams(
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

  protected def setTensorByIndicesToValue(
                                           prevInputIds: Array[Float],
                                           indices: IndexedSeq[Boolean],
                                           value: Float): Array[Float] = {
    for ((inputId, index) <- prevInputIds.zip(indices)) yield if (index) value else inputId
  }

  protected def calcBannedNgramTokens(
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

  protected def tag(
                     batch: Seq[Array[Int]],
                     maxNewTokens: Int,
                     maxTextLength: Int,
                     doSample: Boolean,
                     topK: Int,
                     topP: Double,
                     temperature: Double,
                     noRepeatNgramSize: Int,
                     repetitionPenalty: Double,
                     randomSeed: Option[Long],
                     ignoreTokenIds: Array[Int] = Array(),
                     stopAtEos: Boolean): Array[Array[Int]]

  def predict(
                sentences: Seq[Annotation],
                task: String,
                batchSize: Int,
                maxNewTokens: Int,
                maxTextLength: Int,
                doSample: Boolean,
                topK: Int,
                topP: Double,
                temperature: Double,
                randomSeed: Option[Long] = None,
                ignoreTokenIds: Array[Int] = Array(),
                isCaseSensitive: Boolean,
                stopAtEos: Boolean,
                noRepeatNgramSize: Int,
                repetitionPenalty: Double
              ): Seq[Annotation] = {

    val batchDecoder = sentences.grouped(batchSize).toArray.flatMap { batch =>

      val batchSP = encode(batch, task)
      val spIds = tag(
        batch = batchSP,
        maxNewTokens = maxNewTokens,
        maxTextLength = maxTextLength,
        doSample = doSample,
        topK = topK,
        topP = topP,
        temperature = temperature,
        randomSeed = randomSeed,
        ignoreTokenIds = ignoreTokenIds,
        stopAtEos = stopAtEos,
        noRepeatNgramSize=noRepeatNgramSize,
        repetitionPenalty=repetitionPenalty
      )
      decode(spIds)

    }

    var sentBegin, nextSentEnd = 0
    batchDecoder.zip(sentences).map { case (content, sent) =>
      nextSentEnd += content.length - 1
      val newAnnotation = new Annotation(
        annotatorType = AnnotatorType.DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = sent.metadata)
      sentBegin += nextSentEnd + 1
      newAnnotation
    }
  }

  def generate(
                         prompts: Seq[Annotation],
                         batchSize: Int,
                         maxNewTokens: Int,
                         maxContextLength: Int,
                         doSample: Boolean,
                         topK: Int,
                         topP: Double,
                         temperature: Double,
                         randomSeed: Option[Long],
                         ignoreTokenIds: Array[Int],
                         isCaseSensitive: Boolean,
                         stopAtEos: Boolean,
                         noRepeatNgramSize: Int,
                         repetitionPenalty: Double): Seq[Annotation] = {
    predict(
      sentences = prompts,
      task = "",
      batchSize = batchSize,
      maxNewTokens = maxNewTokens,
      maxTextLength = maxContextLength,
      doSample = doSample,
      topK = topK,
      topP=topP,
      temperature=temperature,
      randomSeed = randomSeed,
      ignoreTokenIds = ignoreTokenIds,
      isCaseSensitive = isCaseSensitive,
      stopAtEos = stopAtEos,
      noRepeatNgramSize = noRepeatNgramSize,
      repetitionPenalty = repetitionPenalty)
  }

  class DecoderProcessor(
                          val batchSize: Int,
                          val maxTextLength: Int,
                          val sequenceLength: Int,
                          val doSample: Boolean,
                          val topK: Int,
                          val topP: Double,
                          val temperature: Double,
                          val vocabSize: Int,
                          val noRepeatNgramSize: Int,
                          val repetitionPenalty: Double,
                          val randomSeed: Option[Long],
                          val stopTokens: Array[Int],
                          val ignoreTokenIds: Array[Int],
                          val maxNewTokens: Int
                        ) {
    var unfinishedSentences: List[Int] = List.fill(batchSize)(1)
    var sentenceLengths: List[Int] = List.fill(batchSize)(maxTextLength)
    var currentLength = sequenceLength
    var nPredictedTokens = 0

    def stopDecoding(decoderInputIds: Array[Array[Int]]): Boolean = {
      // stop when there is a eos in each sentence, or if we exceed the maximum length
      //      stopDecoder = curLen < maxOutputLength || unfinishedSents.max == 0

      (decoderInputIds.forall(o => o exists (t => stopTokens.contains(t)))
        || (nPredictedTokens >= maxNewTokens)
        || (decoderInputIds.head.length > maxTextLength))
    }

    def stopDecoding(decoderInputIds: Array[Array[Long]]): Boolean = {
      stopDecoding(decoderInputIds.map(x => x.map(_.toInt)))
    }

    def processLogits(batchLogits: Array[Array[Float]], decoderInputIds: Array[Array[Long]]): Array[Array[Long]] = {
      processLogits(batchLogits, decoderInputIds.map(x => x.map(_.toInt))).map(x => x.map(_.toLong))
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
        for ((prevInputId, _) <- prevInputIds.zipWithIndex) {
          var logitPenalty = 1.0
          if (logits(i)(prevInputId) < 0) {
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

    private def softmax(values: Array[Float]): Array[Float] = {
      val expElem = values.map(exp(_))
      val total = expElem.sum
      expElem.map(_ / total).map(_.toFloat)
    }

    private def categoricalSample(dist: Array[Float], randomSeed: Option[Long]): Int = {
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
      indices(0)
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

    def processLogits(batchLogits: Array[Array[Float]], decoderInputIds: Array[Array[Int]]): Array[Array[Int]] = {

      nPredictedTokens += 1

      var nextTokenLogits = batchLogits.map(logits => {
        logits.indices
          .map(i => {
            if (ignoreTokenIds.contains(i)) Float.NegativeInfinity else logits(i)
          }).toArray
      })

      // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
      if (repetitionPenalty != 1.0) {
        nextTokenLogits =
          createNextTokenLogitsPenalties(decoderInputIds, nextTokenLogits, repetitionPenalty)
      }

      if (noRepeatNgramSize > 0) {
        // calculate a list of banned tokens to prevent repetitively generating the same ngrams
        // from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        val bannedTokens = calcBannedNgramTokens(decoderInputIds, batchSize, noRepeatNgramSize, currentLength)
        // create bannedTokens boolean mask
        var bannedTokensIndicesMask = Array.empty[IndexedSeq[Boolean]]
        for (bannedTokensSlice <- bannedTokens) {
          bannedTokensIndicesMask = bannedTokensIndicesMask :+
            (for (token <- 0 until vocabSize)
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

      if (randomSeed.isDefined)
        scala.util.Random.setSeed(randomSeed.get)

      val predictions = if (doSample){

        // Temperature (higher temperature => more likely to sample low probability tokens)
        if (temperature != 1.0)
          nextTokenLogits =
            for (nextTokenLogit <- nextTokenLogits)
              yield nextTokenLogit.map(_ / temperature.toFloat)
        // Top-p/top-k filtering
        nextTokenLogits = topKTopPFiltering(nextTokenLogits, topK, topP)
        // Sample

        nextTokenLogits.map(input => categoricalSample(input, randomSeed))
      } else {
        nextTokenLogits.map(x => x.zipWithIndex.maxBy(_._1)._2)
      }
      //      var tokensToAdd = Array.ofDim[Int](decoderInputIds.length)
      val tokensToAdd = predictions.zip(unfinishedSentences).map(x => x._1 * x._2 + paddingTokenId * (1 - x._2))

      currentLength += 1

      val eosInSentences = tokensToAdd.map(x => if (stopTokens.contains(x)) 1 else 0)
      // if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
      val areSentencesUnfinishedAndTokenToAddIsEos =
        unfinishedSentences.zip(eosInSentences).map(x => x._1 * x._2)

      sentenceLengths = sentenceLengths
        .zip(areSentencesUnfinishedAndTokenToAddIsEos)
        .map(x => x._1 * (1 - x._2) + currentLength * x._2)

      // unfinishedSents is set to zero if eos in sentence
      unfinishedSentences =
        unfinishedSentences.zip(areSentencesUnfinishedAndTokenToAddIsEos).map(x => x._1 - x._2)

      decoderInputIds
        .zip(tokensToAdd)
        .map(x => {
          x._1 ++ Array(x._2)
        })

    }
  }

}
