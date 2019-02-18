package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp.annotators.common.{TokenizedSentence, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.annotators.ner.Verbose

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random


class TensorflowNer
(
  val tensorflow: TensorflowWrapper,
  val encoder: NerDatasetEncoder,
  val batchSize: Int,
  override val verboseLevel: Verbose.Value
) extends Logging {

  override def getLogName: String = "NerDL"

  private val charIdsKey = "char_repr/char_ids"
  private val wordLengthsKey = "char_repr/word_lengths"
  private val wordEmbeddingsKey = "word_repr_1/word_embeddings"
  private val sentenceLengthsKey = "word_repr/sentence_lengths"
  private val dropoutKey = "training/dropout"

  private val learningRateKey = "training/lr"
  private val labelsKey = "training/labels"

  private val lossKey = "inference/loss"
  private val trainingKey = "training_1/Momentum"
  private val predictKey = "context_repr/predicted_labels"

  def predict(dataset: Array[WordpieceEmbeddingsSentence]): Array[Array[String]] = {

    val result = ArrayBuffer[Array[String]]()

    for (slice <- dataset.grouped(batchSize)) {
      val batchInput = encoder.encodeInputData(slice)

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
      tensorflow.session.runner.addTarget("training_1/init").run()

    // Train
    for (epoch <- startEpoch until endEpoch) {

      val epochDataset = Random.shuffle(trainDataset.toList).toArray
      val learningRate = lr / (1 + po * epoch)

      log(s"Epoch: $epoch started, learning rate: $learningRate, dataset size: ${epochDataset.length}", Verbose.Epochs)

      val time = System.nanoTime()
      var batches = 0
      var loss = 0f
      for (slice <- epochDataset.grouped(batchSize)) {
        val sentences = slice.map(r => r._2)
        val tags = slice.map(r => r._1.labels.toArray)

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
                  nErrorsToPrint: Int = 0
                 ): Unit = {

    val started = System.nanoTime()

    val predictedCorrect = mutable.Map[String, Int]()
    val predicted = mutable.Map[String, Int]()
    val correct = mutable.Map[String, Int]()

    val sentenceTokens = labeled.map(pair => pair._2.tokens
      .filter(t => t.isWordStart)
      .map(t => t.token)
    ).toList
    val sentenceLabels = labeled.map(pair => pair._1.labels.toArray).toList
    val sentencePredictedTags = labeled.map(pair => predict(Array(pair._2)).head).toList

    var errorsPrinted = 0
    var linePrinted = false
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


