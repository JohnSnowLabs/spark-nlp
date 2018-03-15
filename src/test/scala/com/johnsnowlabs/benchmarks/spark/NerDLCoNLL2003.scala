package com.johnsnowlabs.benchmarks.spark

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence
import com.johnsnowlabs.nlp.annotators.common.{NerTagged, TaggedSentence}
import com.johnsnowlabs.nlp.annotators.ner.{NerConverter, Verbose}
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.datasets.CoNLL
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

import scala.collection.mutable


object NerDLPipeline extends App {
  val folder = "./"

  val trainFile = ExternalResource(folder + "eng.train", ReadAs.LINE_BY_LINE, Map.empty[String, String])
  val testFileA = ExternalResource(folder + "eng.testa", ReadAs.LINE_BY_LINE, Map.empty[String, String])
  val testFileB = ExternalResource(folder + "eng.testb", ReadAs.LINE_BY_LINE, Map.empty[String, String])

  val nerReader = CoNLL(annotatorType = AnnotatorType.NAMED_ENTITY)

  def createPipeline() = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setCustomBoundChars(Array("\n\n"))
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val nerTagger = new NerDLApproach()
      .setInputCols("sentence", "token")
      .setLabelColumn("label")
      .setMaxEpochs(10)
      .setRandomSeed(100)
      .setLr(0.2f)
      .setDropout(0.5f)
      .setOutputCol("ner")
      .setEmbeddingsSource("glove.6B.100d.txt", 100, WordEmbeddingsFormat.TEXT)
      .setVerbose(Verbose.Epochs)

    val converter = new NerConverter()
      .setInputCols("document", "token", "ner")
      .setOutputCol("ner_span")

    Array(documentAssembler,
      sentenceDetector,
      tokenizer,
      nerTagger,
      converter
    )
  }

  def trainNerModel(er: ExternalResource): PipelineModel = {
    System.out.println("Dataset Reading")
    val time = System.nanoTime()
    val dataset = nerReader.readDataset(er, SparkAccessor.benchmarkSpark)
    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")

    System.out.println("Start fitting")

    val stages = createPipeline()

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

    for (label <- notEmptyLabels) {
      val (prec, rec, f1) = calcStat(
        correct.getOrElse(label, 0),
        predicted.getOrElse(label, 0),
        predictedCorrect.getOrElse(label, 0)
      )

      System.out.println(s"$label\t$prec\t$rec\t$f1")
    }
  }

  def measureNer(model: PipelineModel): Unit = {

    System.out.println("\n\nQuality on train data")
    testDataset(trainFile, model, "ner", nerReader, collectNerLabeled)

    System.out.println("\n\nQuality on test A data")
    testDataset(testFileA, model, "ner", nerReader, collectNerLabeled)

    System.out.println("\n\nQuality on test B data")
    testDataset(testFileB, model, "ner", nerReader, collectNerLabeled)
  }

  def getUserFriendly(model: PipelineModel, file: ExternalResource): Array[Array[Annotation]] = {
    val df = model.transform(nerReader.readDataset(file, SparkAccessor.benchmarkSpark))
    Annotation.collect(df, "ner_span")
  }

  val model = trainNerModel(trainFile)
  measureNer(model)

  val annotations = getUserFriendly(model, testFileB)
  NerHelper.saveNerSpanTags(annotations, "predicted.csv")

  model.write.overwrite().save("ner_model")
  PipelineModel.read.load("ner_model")
}

