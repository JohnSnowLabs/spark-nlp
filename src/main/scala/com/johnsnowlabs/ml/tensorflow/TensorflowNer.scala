package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp.annotators.common.WordpieceEmbeddingsSentence
import com.johnsnowlabs.nlp.annotators.ner.Verbose

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Random



class TensorflowNer
(
  val tensorflow: TensorflowWrapper,
  val encoder: NerDatasetEncoder,
  val batchSize: Int,
  override val verboseLevel: Verbose.Value
) extends Logging {

  override def getLogName: String = "NerDL"

  private val charIdsKey = "char_repr/char_ids:0"
  private val wordLengthsKey = "char_repr/word_lengths:0"
  private val wordEmbeddingsKey = "word_repr_1/word_embeddings:0"
  private val sentenceLengthsKey = "word_repr/sentence_lengths:0"
  private val dropoutKey = "training/dropout:0"

  private val learningRateKey = "training/lr:0"
  private val labelsKey = "training/labels:0"

  private val lossKey = "inference/Mean:0"
  private val trainingKey = "training_1/Adam"
  private val predictKey = "inference/cond_2/Merge:0"

  private val initKey = "training_1/init"

  def doSlice[T: ClassTag](dataset: TraversableOnce[T], getLen: T => Int, batchSize: Int = 32): Iterator[Array[T]] = {
    val gr = SentenceGrouper[T](getLen)
    gr.slice(dataset, batchSize)
  }

  def slice(dataset: TraversableOnce[(TextSentenceLabels, WordpieceEmbeddingsSentence)], batchSize: Int = 32):
      Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] = {
    doSlice[(TextSentenceLabels, WordpieceEmbeddingsSentence)](dataset, _._2.tokens.length, batchSize)
  }

  def predict(dataset: Array[WordpieceEmbeddingsSentence]): Array[Array[String]] = {

    val result = ArrayBuffer[Array[String]]()

    for (batch <- dataset.grouped(batchSize); if batch.length > 0) {
      val batchInput = encoder.encodeInputData(batch)

      if (batchInput.sentenceLengths.length == 0)
        for (_ <- batch) {
          result.append(Array.empty[String])
        }
      else {
        val tensors = new TensorResources()

        val calculated = tensorflow.session.runner
          .feed(sentenceLengthsKey, tensors.createTensor(batchInput.sentenceLengths))
          .feed(wordEmbeddingsKey, tensors.createTensor(batchInput.wordEmbeddings))

          .feed(wordLengthsKey, tensors.createTensor(batchInput.wordLengths))
          .feed(charIdsKey, tensors.createTensor(batchInput.charIds))

          .feed(dropoutKey, tensors.createTensor(1.0f))
          .fetch(predictKey)
          .run()

        tensors.clearTensors()

        val tagIds = TensorResources.extractInts(calculated.get(0))
        val tags = encoder.decodeOutputData(tagIds)
        val sentenceTags = encoder.convertBatchTags(tags, batchInput.sentenceLengths)

        result.appendAll(sentenceTags)
      }
    }

    result.toArray
  }

  def train(trainDataset: Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)],
            lr: Float,
            po: Float,
            batchSize: Int,
            dropout: Float,
            startEpoch: Int,
            endEpoch: Int,
            validation: Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)] = Array.empty
           ): Unit = {

    log(s"Training started, trainExamples: ${trainDataset.length}, " +
      s"labels: ${encoder.tags.length} " +
      s"chars: ${encoder.chars.length}, ", Verbose.TrainingStat)

    // Initialize
    if (startEpoch == 0)
      tensorflow.session.runner.addTarget(initKey).run()

    val trainDatasetSeq = trainDataset.toSeq
    // Train
    for (epoch <- startEpoch until endEpoch) {

      val epochDataset = Random.shuffle(trainDatasetSeq)
      val learningRate = lr / (1 + po * epoch)

      log(s"Epoch: $epoch started, learning rate: $learningRate, dataset size: ${epochDataset.length}", Verbose.Epochs)

      val time = System.nanoTime()
      var batches = 0
      var loss = 0f
      for (batch <- slice(epochDataset, batchSize)) {
        val sentences = batch.map(r => r._2)
        val tags = batch.map(r => r._1.labels.toArray)

        val batchInput = encoder.encodeInputData(sentences)
        val batchTags = encoder.encodeTags(tags)

        val tensors = new TensorResources()
        val calculated = tensorflow.session.runner
          .feed(sentenceLengthsKey, tensors.createTensor(batchInput.sentenceLengths))
          .feed(wordEmbeddingsKey, tensors.createTensor(batchInput.wordEmbeddings))

          .feed(wordLengthsKey, tensors.createTensor(batchInput.wordLengths))
          .feed(charIdsKey, tensors.createTensor(batchInput.charIds))
          .feed(labelsKey, tensors.createTensor(batchTags))

          .feed(dropoutKey, tensors.createTensor(dropout))
          .feed(learningRateKey, tensors.createTensor(learningRate))

          .fetch(lossKey)
          .addTarget(trainingKey)
          .run()

        loss += calculated.get(0).floatValue()

        tensors.clearTensors()
        batches += 1
      }

      log(s"Done, ${(System.nanoTime() - time)/1e9} loss: $loss, batches: $batches", Verbose.Epochs)

      if (validation.nonEmpty) {
        log("Quality on train dataset: ", Verbose.Epochs)
        measure(trainDataset, (s: String) => log(s, Verbose.Epochs))
      }

      if (validation.nonEmpty) {
        log("Quality on validation dataset: ", Verbose.Epochs)
        measure(validation, (s: String) => log(s, Verbose.Epochs))
      }
    }
  }


  def calcStat(correct: Int, predicted: Int, predictedCorrect: Int): (Float, Float, Float) = {
    // prec = (predicted & correct) / predicted
    // rec = (predicted & correct) / correct
    val prec = predictedCorrect.toFloat / predicted
    val rec = predictedCorrect.toFloat / correct
    val f1 = 2 * prec * rec / (prec + rec)

    (prec, rec, f1)
  }

  def measure(labeled: Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)],
                  log: (String => Unit),
                  extended: Boolean = false,
                  nErrorsToPrint: Int = 0,
                  batchSize: Int = 20
                 ): Unit = {

    val started = System.nanoTime()

    val predictedCorrect = mutable.Map[String, Int]()
    val predicted = mutable.Map[String, Int]()
    val correct = mutable.Map[String, Int]()

    var errorsPrinted = 0
    var linePrinted = false

    for (batch <- slice(labeled, batchSize)) {

      val sentenceTokens = batch.map(pair => pair._2.tokens
        .filter(t => t.isWordStart)
        .map(t => t.token)
      ).toList
      val sentenceLabels = batch.map(pair => pair._1.labels.toArray).toList
      val sentencePredictedTags = predict(batch.map(_._2))

      (sentenceTokens, sentenceLabels, sentencePredictedTags).zipped.foreach {
        case (tokens, labels, tags) =>
          for (i <- 0 until labels.length) {
            val label = labels(i)
            val tag = tags(i)
            val iWord = tokens(i)

            correct(label) = correct.getOrElse(label, 0) + 1
            predicted(tag) = predicted.getOrElse(tag, 0) + 1

            if (label == tag)
              predictedCorrect(tag) = predictedCorrect.getOrElse(tag, 0) + 1
            else if (errorsPrinted < nErrorsToPrint) {
              log(s"label: $label, predicted: $tag, word: $iWord")
              linePrinted = false
              errorsPrinted += 1
            }
          }

          if (errorsPrinted < nErrorsToPrint && !linePrinted) {
            log("")
            linePrinted = true
          }
      }
    }

    if (extended)
      log(s"time: ${(System.nanoTime() - started)/1e9}")

    val labels = (correct.keys ++ predicted.keys).toSeq.distinct

    val notEmptyLabels = labels.filter(label => label != "O" && label.nonEmpty)

    val totalCorrect = correct.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalPredicted = predicted.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalPredictedCorrect = predictedCorrect.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val (prec, rec, f1) = calcStat(totalCorrect, totalPredicted, totalPredictedCorrect)
    log(s"Total stat, prec: $prec\t, rec: $rec\t, f1: $f1")

    if (!extended)
      return

    log("label\tprec\trec\tf1")

    for (label <- notEmptyLabels) {
      val (prec, rec, f1) = calcStat(
        correct.getOrElse(label, 0),
        predicted.getOrElse(label, 0),
        predictedCorrect.getOrElse(label, 0)
      )

      log(s"$label\t$prec\t$rec\t$f1")
    }
  }
}
