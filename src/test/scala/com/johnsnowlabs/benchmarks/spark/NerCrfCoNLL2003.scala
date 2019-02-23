package com.johnsnowlabs.benchmarks.spark

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.johnsnowlabs.nlp.annotators.common.{NerTagged, PosTagged, TaggedSentence}
import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproachDistributed
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.datasets.CoNLL
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.{PipelineModel, PipelineStage}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable


object CoNLL2003PipelineTest extends App {
  val folder = "./"

  val trainFile = ExternalResource(folder + "eng.train", ReadAs.LINE_BY_LINE, Map("delimiter" -> " "))
  val testFileA = ExternalResource(folder + "eng.testa", ReadAs.LINE_BY_LINE, Map("delimiter" -> " "))
  val testFileB = ExternalResource(folder + "eng.testb", ReadAs.LINE_BY_LINE, Map("delimiter" -> " "))

  val nerReader = CoNLL(annotatorType = AnnotatorType.NAMED_ENTITY)
  val posReader = CoNLL(targetColumn = 1, annotatorType = AnnotatorType.POS)

  def getPosStages(): Array[_ <: PipelineStage] = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setCustomBounds(Array(System.lineSeparator()*2))
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val posTagger = new PerceptronApproachDistributed()
      .setNIterations(10)
      .setInputCols("token", "document")
      .setOutputCol("pos")

    Array(documentAssembler,
      sentenceDetector,
      tokenizer,
      posTagger)
  }

  def getNerStages(): Array[_ <: PipelineStage] = {

    val nerTagger = new NerCrfApproach()
      .setInputCols("sentence", "token", "pos")
      .setLabelColumn("label")
      .setExternalFeatures(ExternalResource("eng.train", ReadAs.LINE_BY_LINE, Map("delimiter" -> " ")))
      .setC0(2250000)
      .setRandomSeed(100)
      .setMaxEpochs(10)
      .setOutputCol("ner")
      .setEmbeddingsSource("glove.6B.100d.txt", 100, WordEmbeddingsFormat.TEXT)

    getPosStages() :+ nerTagger
  }

  def trainPosModel(er: ExternalResource): PipelineModel = {
    System.out.println("Dataset Reading")
    val time = System.nanoTime()
    val dataset = posReader.readDataset(er, SparkAccessor.spark)
    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")

    System.out.println("Start fitting")

    val stages = getPosStages()

    val pipeline = new RecursivePipeline()
      .setStages(stages)

    pipeline.fit(dataset)
  }

  def trainNerModel(er: ExternalResource): PipelineModel = {
    System.out.println("Dataset Reading")
    val time = System.nanoTime()
    val dataset = nerReader.readDataset(er, SparkAccessor.benchmarkSpark)
    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")

    System.out.println("Start fitting")

    val stages = getNerStages()

    val pipeline = new RecursivePipeline()
      .setStages(stages)

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

  def collectNerLabeled(df: DataFrame): Seq[(TextSentenceLabels, NerTaggedSentence)] = {
    NerTagged.collectLabeledInstances(
      df,
      Seq("sentence", "token", "ner"),
      "label"
    )
  }

  def collectPosLabeled(df: DataFrame): Seq[(TextSentenceLabels, PosTaggedSentence)] = {
    PosTagged.collectLabeledInstances(
      df,
      Seq("sentence", "token", "pos"),
      "label"
    )
  }

  def testDataset(er: ExternalResource,
                  model: PipelineModel,
                  predictedColumn: String = "ner",
                  reader: CoNLL,
                  collect: DataFrame => Seq[(TextSentenceLabels, TaggedSentence)]
                 ): Unit = {
    val started = System.nanoTime()

    val predictedCorrect = mutable.Map[String, Int]()
    val predicted = mutable.Map[String, Int]()
    val correct = mutable.Map[String, Int]()

    val dataset = reader.readDataset(er, SparkAccessor.benchmarkSpark)
    val transformed = model.transform(dataset)

    val sentences = collect(transformed)

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
    val totalPredicted = predicted.filterKeys(label => notEmptyLabels.contains(label)).values.sum
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

  def measurePos(): PipelineModel = {
    val model = trainPosModel(trainFile)

    System.out.println("\n\nQuality on train data")
    testDataset(trainFile, model, "pos", posReader, collectPosLabeled)

    System.out.println("\n\nQuality on test A data")
    testDataset(testFileA, model, "pos", posReader, collectPosLabeled)

    System.out.println("\n\nQuality on test B data")
    testDataset(testFileB, model, "pos", posReader, collectPosLabeled)

    model
  }

  def measureNer(): PipelineModel = {
    val model = trainNerModel(trainFile)

    System.out.println("\n\nQuality on train data")
    testDataset(trainFile, model, "ner", nerReader, collectNerLabeled)

    System.out.println("\n\nQuality on test A data")
    testDataset(testFileA, model, "ner", nerReader, collectNerLabeled)

    System.out.println("\n\nQuality on test B data")
    testDataset(testFileB, model, "ner", nerReader, collectNerLabeled)

    model
  }

  val posModel = measurePos()
  posModel.write.overwrite().save("pos_model")

  val nerModel = measureNer()
  nerModel.write.overwrite().save("ner_model")
}
