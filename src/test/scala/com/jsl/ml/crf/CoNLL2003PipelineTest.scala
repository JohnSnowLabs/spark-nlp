package com.jsl.ml.crf

import com.jsl.nlp._
import com.jsl.nlp.annotators.RegexTokenizer
import com.jsl.nlp.annotators.ner.crf.{CrfBasedNer, NerTagged}
import com.jsl.nlp.annotators.pos.perceptron.PerceptronApproach
import com.jsl.nlp.annotators.sbd.pragmatic.SentenceDetectorModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Dataset
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object CoNLL
{
  /*
    Reads Dataset in CoNLL format and pack it into docs
   */
  def readDocs(file: String): Iterator[(String, Seq[Annotation])] = {
    val doc = new StringBuilder()
    val labels = new ArrayBuffer[Annotation]()

    val lines = Source.fromFile(file).getLines()

    val docs = lines
      .flatMap{line =>
        val items = line.split(" ")
        val word = items(0)
        if (word == "-DOCSTART-") {
          val result = (doc.toString, labels.toList)
          doc.clear()
          labels.clear()

          if (result._1.nonEmpty)
            Some(result)
          else
            None
        } else if (items.length == 1) {
          doc.append("\n")
          None
        } else
        {
          if (doc.nonEmpty)
            doc.append(" ")

          val begin = doc.length
          doc.append(word)
          val end = doc.length - 1
          val ner = items(3)
          labels.append(new Annotation(AnnotatorType.NAMED_ENTITY, begin, end, Map("tag" -> ner)))
          None
        }
      }

    val last = if (doc.nonEmpty) Seq((doc.toString, labels.toList)) else Seq.empty

    docs ++ last
  }
}

object CoNLL2003PipelineTest extends App {
  val folder = "./"

  val trainFile = folder + "eng.train"
  val testFileA = folder + "eng.testa"
  val testFileB = folder + "eng.testb"

  def readDataset(file: String, textColumn: String = "text", labelColumn: String = "label"): Dataset[_] = {
    val seq = CoNLL.readDocs(file).toSeq

    import SparkAccessor.spark.implicits._

    seq.toDF(textColumn, labelColumn)
  }

  def trainModel(file: String): PipelineModel = {
    System.out.println("Dataset Reading")
    val time = System.nanoTime()
    val dataset = readDataset(file)
    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")

    System.out.println("Start fitting")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetectorModel()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new RegexTokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val posTagger = new PerceptronApproach()
      .setCorpusPath("/anc-pos-corpus/")
      .setNIterations(5)
      .setInputCols("token", "document")
      .setOutputCol("pos")

    val nerTagger = new CrfBasedNer()
      .setInputCols("sentence", "token", "pos")
      .setLabelColumn("label")
      .setOutputCol("ner")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        nerTagger
      ))

    pipeline.fit(dataset)
  }

  def calcStat(correct: Int, predicted: Int, predictedCorrect: Int): (Float, Float, Float) = {
    // prec = (predicted & correct) / predicted
    // rec = (predicted & correct) / correct
    val prec = predictedCorrect.toFloat / predicted
    val rec = predictedCorrect.toFloat / correct
    val f1 = 2 * prec * rec / (prec + rec)

    (prec, rec, f1)
  }

  def testDataset(file: String, model: PipelineModel): Unit = {
    val started = System.nanoTime()

    val predictedCorrect = mutable.Map[String, Int]()
    val predicted = mutable.Map[String, Int]()
    val correct = mutable.Map[String, Int]()

    val dataset = readDataset(file)

    val transformed = model.transform(dataset)

    val sentences = NerTagged.collectNerInstances(
      transformed,
      Seq("sentence", "token", "ner"),
      "label"
    )

    sentences.foreach{
      case (labels, taggedSentence) =>
        labels.labels.zip(taggedSentence.tags).foreach {
          case (label, tag) =>
            correct(label) = correct.getOrElse(label, 0) + 1
            predicted(tag) = predicted.getOrElse(tag, 0) + 1

            if (label == tag)
              predictedCorrect(tag) = predictedCorrect.getOrElse(tag, 0) + 1
        }
    }

    System.out.println(s"time: ${(System.nanoTime() - started)/1e9}")

    val labels = (correct.keys ++ predicted.keys).toSeq.distinct

    val notEmptyLabels = labels.filter(label => label != "O" && label.nonEmpty)

    val totalCorrect = correct.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalPredicted = correct.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalPredictedCorrect = predictedCorrect.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val (prec, rec, f1) = calcStat(totalCorrect, totalPredicted, totalPredictedCorrect)
    System.out.println(s"Total stat, prec: $prec\t, rec: $rec\t, f1: $f1")

    System.out.println("label\tprec\trec\tf1")

    for (label <- labels) {
      val (prec, rec, f1) = calcStat(
        correct.getOrElse(label, 0),
        predicted.getOrElse(label, 0),
        predictedCorrect.getOrElse(label, 0)
      )

      System.out.println(s"$label\t$prec\t$rec\t$f1")
    }
  }

  val model = trainModel(trainFile)

  System.out.println("\n\nQuality on train data")
  testDataset(trainFile, model)

  System.out.println("\n\nQuality on test A data")
  testDataset(testFileA, model)

  System.out.println("\n\nQuality on test B data")
  testDataset(testFileB, model)
}
