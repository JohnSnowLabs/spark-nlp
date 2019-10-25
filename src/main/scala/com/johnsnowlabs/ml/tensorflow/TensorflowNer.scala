package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp.annotators.common.WordpieceEmbeddingsSentence
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import org.apache.spark.ml.util.Identifiable

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
  private val scoresKey = "context_repr/scores:0"
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

  def predict(
               dataset: Array[WordpieceEmbeddingsSentence],
               configProtoBytes: Option[Array[Byte]] = None,
               includeConfidence: Boolean = false): Array[Array[(String, Option[Double])]] = {

    val result = ArrayBuffer[Array[(String, Option[Double])]]()

    for (batch <- dataset.grouped(batchSize); if batch.length > 0) {
      val batchInput = encoder.encodeInputData(batch)

      if (batchInput.sentenceLengths.length == 0)
        for (_ <- batch) {
          result.append(Array.empty[(String, Option[Double])])
        }
      else {
        val tensors = new TensorResources()

        val calculator = tensorflow.getSession(configProtoBytes=configProtoBytes).runner
          .feed(sentenceLengthsKey, tensors.createTensor(batchInput.sentenceLengths))
          .feed(wordEmbeddingsKey, tensors.createTensor(batchInput.wordEmbeddings))

          .feed(wordLengthsKey, tensors.createTensor(batchInput.wordLengths))
          .feed(charIdsKey, tensors.createTensor(batchInput.charIds))

          .feed(dropoutKey, tensors.createTensor(1.0f))
          .fetch(predictKey)

        val calculatorInc = if (includeConfidence) calculator.fetch(scoresKey) else calculator

        val calculated = calculatorInc.run()

        tensors.clearTensors()

        val tagIds = TensorResources.extractInts(calculated.get(0))

        val confidence: Option[Seq[Double]] = {
          if (includeConfidence) {
            val scores = TensorResources.extractFloats(calculated.get(1))
            require(scores.length % tagIds.length == 0, "tag size mismatch against feed size. please report an issue.")
            val exp = scores.map(s => math.exp(s.toDouble)).grouped(scores.length / tagIds.length).toSeq
            val probs = exp.map(g => BigDecimal(g.map(_ / g.sum).max).setScale(4, BigDecimal.RoundingMode.HALF_UP).toDouble)
            Some(probs)
          } else {
            None
          }
        }

        val tags = encoder.decodeOutputData(tagIds)
        val sentenceTags = encoder.convertBatchTags(tags, batchInput.sentenceLengths, confidence)

        result.appendAll(sentenceTags)
      }
    }

    result.toArray
  }

  def getPiecesTags(tokenTags: TextSentenceLabels, sentence: WordpieceEmbeddingsSentence): Array[String] = {
    var i = -1

    sentence.tokens.map{t =>
      i += 1
      tokenTags.labels(i)
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
            validationSplit: Float = 0.0f,
            evaluationLogExtended: Boolean = false,
            includeConfidence: Boolean = false,
            enableOutputLogs: Boolean = false,
            uuid: String  = Identifiable.randomUID("annotator")
           ): Unit = {

    log(s"Name of the selected graph: $graphFileName", Verbose.Epochs)
    outputLog(s"Name of the selected graph: $graphFileName", uuid, enableOutputLogs)

    // Initialize
    if (startEpoch == 0)
      tensorflow.createSession(configProtoBytes=configProtoBytes).runner.addTarget(initKey).run()

    val sample: Int = (trainDataset.length*validationSplit).toInt

    val (trainDatasetSeq, validateDatasetSample) = if (validationSplit > 0f) {
      val (trainingSample, trainingSet) = Random.shuffle(trainDataset.toSeq).splitAt(sample)
      (trainingSet, trainingSample.toArray)
    } else {
      // No validationSplit has been set so just use the entire training Dataset
      val emptyValid: Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)] = Array.empty
      (trainDataset.toSeq, emptyValid)
    }

    log(s"Training started, trainExamples: ${trainDatasetSeq.length}, " +
      s"labels: ${encoder.tags.length} " +
      s"chars: ${encoder.chars.length}, ", Verbose.TrainingStat)

    outputLog(s"Training started, trainExamples: ${trainDatasetSeq.length}, " +
      s"labels: ${encoder.tags.length} " +
      s"chars: ${encoder.chars.length}, ", uuid, enableOutputLogs)

    // Train
    for (epoch <- startEpoch until endEpoch) {

      val epochDataset = Random.shuffle(trainDatasetSeq)
      val learningRate = lr / (1 + po * epoch)

      log(s"Epoch: $epoch started, learning rate: $learningRate, dataset size: ${epochDataset.length}", Verbose.Epochs)
      outputLog("\n", uuid, enableOutputLogs)
      outputLog(s"Epoch: $epoch started, learning rate: $learningRate, dataset size: ${epochDataset.length}", uuid, enableOutputLogs)

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
      outputLog(s"Done, ${(System.nanoTime() - time)/1e9} loss: $loss, batches: $batches", uuid, enableOutputLogs)

      if (validationSplit > 0.0) {
        log(s"Quality on validation dataset (${validationSplit*100}%), valExamples = $sample", Verbose.Epochs)
        outputLog(s"Quality on validation dataset (${validationSplit*100}%), valExamples = $sample", uuid, enableOutputLogs)
        measure(validateDatasetSample, (s: String) => log(s, Verbose.Epochs), extended = evaluationLogExtended, includeConfidence = includeConfidence, enableOutputLogs = enableOutputLogs, uuid = uuid)
      }

      if (test.nonEmpty) {
        log("Quality on test dataset: ", Verbose.Epochs)
        outputLog("Quality on test dataset: ", uuid, enableOutputLogs)
        measure(test, (s: String) => log(s, Verbose.Epochs), extended = evaluationLogExtended, includeConfidence = includeConfidence, enableOutputLogs = enableOutputLogs, uuid = uuid)
      }

    }
  }

  def calcStat(tp: Int, fp: Int, fn: Int): (Float, Float, Float) = {
    val prec = tp.toFloat / (tp.toFloat + fp.toFloat)
    val rec = tp.toFloat / (tp.toFloat + fn.toFloat)
    val f1 = 2 * ((prec * rec) / (prec + rec))

    (if (prec.isNaN) 0f else prec, if(rec.isNaN) 0f else rec, if (f1.isNaN) 0 else f1)
  }

  def tagsForTokens(labels: Array[(String, Option[Double])], pieces: WordpieceEmbeddingsSentence): Array[(String, Option[Double])] = {
    labels.zip(pieces.tokens).flatMap{
      case(l, p) =>
        if (p.isWordStart)
          Some(l)
        else
          None
    }
  }

  def tagsForTokens(labels: Array[Array[(String, Option[Double])]], pieces: Array[WordpieceEmbeddingsSentence]):
  Array[Array[(String, Option[Double])]] = {

    labels.zip(pieces)
      .map{case (l, p) => tagsForTokens(l, p)}
  }

  def measure(labeled: Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)],
              log: String => Unit,
              extended: Boolean = false,
              batchSize: Int = 100,
              includeConfidence: Boolean = false,
              enableOutputLogs: Boolean = false,
              uuid: String = Identifiable.randomUID("annotator")
             ): Unit = {

    val started = System.nanoTime()

    val predicted = mutable.Map[String, Int]()
    val correct = mutable.Map[String, Int]()

    val truePositives = mutable.Map[String, Int]()
    val falsePositives = mutable.Map[String, Int]()
    val falseNegatives = mutable.Map[String, Int]()

    for (batch <- slice(labeled, batchSize)) {

      val sentencePredictedTags = predict(batch.map(_._2), includeConfidence = includeConfidence)

      val sentenceTokenTags = tagsForTokens(sentencePredictedTags, batch.map(_._2))

      val sentenceTokens = batch.map(pair => pair._2.tokens
        .filter(t => t.isWordStart)
        .map(t => t.token)
      ).toList

      val sentenceLabels = batch.map(pair => pair._1.labels.toArray).toList

      (sentenceTokens, sentenceLabels, sentenceTokenTags).zipped.foreach {
        case (tokens, labels, tags) =>
          for (i <- labels.indices) {

            val label = labels(i)
            val tag = tags(i)
            val iWord = tokens(i)

            correct(label) = correct.getOrElse(label, 0) + 1
            predicted(tag._1) = predicted.getOrElse(tag._1, 0) + 1

            //We don't really care about true negatives at the moment
            if ((label == tag._1)) {
              truePositives(label) = truePositives.getOrElse(label, 0) + 1
            } else if (label == "O" && tag._1 != "O") {
              falsePositives(tag._1) = falsePositives.getOrElse(tag._1, 0) + 1
            } else {
              falsePositives(tag._1) = falsePositives.getOrElse(tag._1, 0) + 1
              falseNegatives(label) = falseNegatives.getOrElse(label, 0) + 1
            }

          }
      }
    }

    log(s"time to finish evaluation: ${(System.nanoTime() - started)/1e9}")
    outputLog(s"time to finish evaluation: ${(System.nanoTime() - started)/1e9}", uuid, enableOutputLogs)

    val labels = (correct.keys ++ predicted.keys).filter(label => label != "O").toSeq.distinct
    val notEmptyLabels = labels.filter(label => label != "O" && label.nonEmpty)

    val totalTruePositives = truePositives.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalFalsePositives = falsePositives.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalFalseNegatives = falseNegatives.filterKeys(label => notEmptyLabels.contains(label)).values.sum

    val (prec, rec, f1) = calcStat(totalTruePositives, totalFalsePositives, totalFalseNegatives)

    if (extended) {
      log("label\t tp\t fp\t fn\t prec\t rec\t f1")
      outputLog("label\t tp\t fp\t fn\t prec\t rec\t f1", uuid, enableOutputLogs)
    }

    var totalPercByClass, totalRecByClass = 0f
    for (label <- labels) {
      val tp = truePositives.getOrElse(label, 0)
      val fp = falsePositives.getOrElse(label, 0)
      val fn = falseNegatives.getOrElse(label, 0)
      val (prec, rec, f1) = calcStat(tp, fp, fn)
      if (extended) {
        log(s"$label\t $tp\t $fp\t $fn\t $prec\t $rec\t $f1")
        outputLog(s"$label\t $tp\t $fp\t $fn\t $prec\t $rec\t $f1", uuid, enableOutputLogs)
      }
      totalPercByClass = totalPercByClass + prec
      totalRecByClass = totalRecByClass + rec
    }
    val macroPercision = totalPercByClass/notEmptyLabels.length
    val macroRecall = totalRecByClass/notEmptyLabels.length
    val macroF1 = 2 * ((macroPercision * macroRecall) / (macroPercision + macroRecall))

    if (extended) {
      log(s"tp: $totalTruePositives fp: $totalFalsePositives fn: $totalFalseNegatives labels: ${notEmptyLabels.length}")
      outputLog(s"tp: $totalTruePositives fp: $totalFalsePositives fn: $totalFalseNegatives labels: ${notEmptyLabels.length}", uuid, enableOutputLogs)
    }
    // ex: Precision = P1+P2/2
    log(s"Macro-average\t prec: $macroPercision, rec: $macroRecall, f1: $macroF1")
    outputLog(s"Macro-average\t prec: $macroPercision, rec: $macroRecall, f1: $macroF1", uuid, enableOutputLogs )
    // ex: Precision =  TP1+TP2/TP1+TP2+FP1+FP2
    log(s"Micro-average\t prec: $prec, rec: $rec, f1: $f1")
    outputLog(s"Micro-average\t prec: $prec, rec: $rec, f1: $f1", uuid, enableOutputLogs)
  }
}
