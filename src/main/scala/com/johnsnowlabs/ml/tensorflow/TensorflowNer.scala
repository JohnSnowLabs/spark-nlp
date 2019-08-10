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
) extends Serializable with Logging {

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

  def predict(dataset: Array[WordpieceEmbeddingsSentence], configProtoBytes: Option[Array[Byte]] = None): Array[Array[String]] = {

    val result = ArrayBuffer[Array[String]]()

    for (batch <- dataset.grouped(batchSize); if batch.length > 0) {
      val batchInput = encoder.encodeInputData(batch)

      if (batchInput.sentenceLengths.length == 0)
        for (_ <- batch) {
          result.append(Array.empty[String])
        }
      else {
        val tensors = new TensorResources()

        val calculated = tensorflow.getSession(configProtoBytes=configProtoBytes).runner
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

  def getPiecesTags(tokenTags: TextSentenceLabels, sentence: WordpieceEmbeddingsSentence): Array[String] = {
    var i = -1

    sentence.tokens.map{t =>
      //if (t.isWordStart) {
      i += 1
      tokenTags.labels(i)
      //}
      //else
      //"X"
    }
  }

  def getPiecesTags(tokenTags: Array[TextSentenceLabels], sentences: Array[WordpieceEmbeddingsSentence])
  :Array[Array[String]] = {
    tokenTags.zip(sentences).map{
      case (tags, sentence) => getPiecesTags(tags, sentence)
    }
  }

  def train(trainDataset: Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)],
            lr: Float,
            po: Float,
            batchSize: Int,
            dropout: Float,
            startEpoch: Int = 0,
            endEpoch: Int,
            graphFileName: String = "",
            test: Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)] = Array.empty,
            configProtoBytes: Option[Array[Byte]] = None,
            trainValidationProp: Float = 0.0f,
            evaluationLogExtended: Boolean = false
           ): Unit = {

    log(s"Name of the selected graph: $graphFileName", Verbose.Epochs)

    log(s"Training started, trainExamples: ${trainDataset.length}, " +
      s"labels: ${encoder.tags.length} " +
      s"chars: ${encoder.chars.length}, ", Verbose.TrainingStat)

    // Initialize
    if (startEpoch == 0)
      tensorflow.createSession(configProtoBytes=configProtoBytes).runner.addTarget(initKey).run()

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
        val tags = getPiecesTags(batch.map(r => r._1), sentences)

        val batchInput = encoder.encodeInputData(sentences)
        val batchTags = encoder.encodeTags(tags)

        val tensors = new TensorResources()
        val calculated = tensorflow.getSession(configProtoBytes=configProtoBytes).runner
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

      if (trainValidationProp > 0.0) {
        val sample: Int = (trainDataset.length*trainValidationProp).toInt

        val trainDatasetSample = trainDataset.take(sample)

        log(s"Quality on training dataset (${trainValidationProp*100}%), trainExamples = $sample", Verbose.Epochs)
        measure(trainDatasetSample, (s: String) => log(s, Verbose.Epochs), extended = evaluationLogExtended)
      }

      if (test.nonEmpty) {
        log("Quality on test dataset: ", Verbose.Epochs)
        measure(test, (s: String) => log(s, Verbose.Epochs), extended = evaluationLogExtended)
      }

    }
  }

  def calcStat(tp: Int, fp: Int, fn: Int): (Float, Float, Float) = {
    val prec = tp.toFloat / (tp.toFloat + fp.toFloat)
    val rec = tp.toFloat / (tp.toFloat + fn.toFloat)
    val f1 = 2 * ((prec * rec) / (prec + rec))

    (if (prec.isNaN) 0f else prec, if(rec.isNaN) 0f else rec, if (f1.isNaN) 0 else f1)
  }

  def tagsForTokens(labels: Array[String], pieces: WordpieceEmbeddingsSentence): Array[String] = {
    labels.zip(pieces.tokens).flatMap{
      case(l, p) =>
        if (p.isWordStart)
          Some(l)
        else
          None
    }
  }

  def tagsForTokens(labels: Array[Array[String]], pieces: Array[WordpieceEmbeddingsSentence]):
  Array[Array[String]] = {

    labels.zip(pieces)
      .map{case (l, p) => tagsForTokens(l, p)}
  }

  def measure(labeled: Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)],
              log: String => Unit,
              extended: Boolean = false,
              batchSize: Int = 100
             ): Unit = {

    val started = System.nanoTime()

    val predicted = mutable.Map[String, Int]()
    val correct = mutable.Map[String, Int]()

    val truePositives = mutable.Map[String, Int]()
    val falsePositives = mutable.Map[String, Int]()
    val falseNegatives = mutable.Map[String, Int]()

    for (batch <- slice(labeled, batchSize)) {

      val sentencePredictedTags = predict(batch.map(_._2))

      val sentenceTokenTags = tagsForTokens(sentencePredictedTags, batch.map(_._2))

      val sentenceTokens = batch.map(pair => pair._2.tokens
        .filter(t => t.isWordStart)
        .map(t => t.token)
      ).toList

      val sentenceLabels = batch.map(pair => pair._1.labels.toArray).toList

      (sentenceTokens, sentenceLabels, sentenceTokenTags).zipped.foreach {
        case (tokens, labels, tags) =>
          for (i <- 0 until labels.length) {

            val label = labels(i)
            val tag = tags(i)
            val iWord = tokens(i)

            correct(label) = correct.getOrElse(label, 0) + 1
            predicted(tag) = predicted.getOrElse(tag, 0) + 1

            //We don't really care about true negatives at the moment
            if ((label == tag) && label != "O") {
              truePositives(label) = truePositives.getOrElse(label, 0) + 1
            } else if (label == "O" && tag != "O") {
              falsePositives(tag) = falsePositives.getOrElse(tag, 0) + 1
            } else {
              falsePositives(tag) = falsePositives.getOrElse(tag, 0) + 1
              falseNegatives(label) = falseNegatives.getOrElse(label, 0) + 1
            }

          }
      }
    }

    log(s"time to finish evaluation: ${(System.nanoTime() - started)/1e9}")

    val labels = (correct.keys ++ predicted.keys).filter(label => label != "O").toSeq.distinct
    val notEmptyLabels = labels.filter(label => label != "O" && label.nonEmpty)

    val totalTruePositives = truePositives.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalFalsePositives = falsePositives.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalFalseNegatives = falseNegatives.filterKeys(label => notEmptyLabels.contains(label)).values.sum

    val (prec, rec, f1) = calcStat(totalTruePositives, totalFalsePositives, totalFalseNegatives)

    if (extended) {
      log("label\t prec\t rec\t f1")
    }

    for (label <- labels) {
      val (prec, rec, f1) = calcStat(
        truePositives.getOrElse(label, 0),
        falsePositives.getOrElse(label, 0),
        falseNegatives.getOrElse(label, 0)
      )
      if (extended) {
        log(s"$label\t $prec\t $rec\t $f1")
      }
    }
    log(s"Total labels in evaluation: ${notEmptyLabels.length}")

    log(s"Weighted stats\t prec: $prec, rec: $rec, f1: $f1")
  }
}
