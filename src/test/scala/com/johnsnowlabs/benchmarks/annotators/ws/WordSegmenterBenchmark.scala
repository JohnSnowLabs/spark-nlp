package com.johnsnowlabs.benchmarks.annotators.ws

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{WordSegmenterApproach, WordSegmenterModel}
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Encoders}
import org.scalatest.FlatSpec

import scala.collection.mutable

class WordSegmenterBenchmark extends FlatSpec {

  private val trainingDataSetFile = "src/test/resources/word-segmenter/chinese_train.utf8"
  private val testingDataSetFile = "src/test/resources/word-segmenter/chinese_test.utf8"
  private val groundTruthDataSetFile = "src/test/resources/word-segmenter/chinese_ground_truth.utf8"

  private val nIterationsList = List(3, 5, 7)
  private val frequencyThresholdList = List(10, 20, 30)
  private val ambiguityThresholdList = List(0.5, 0.70, 0.97, 0.99)

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  "WordSegmenterBenchmark with a set of parameters" should "output metrics" ignore {
    var accuracyByParameters: List[AccuracyByParameter] = List()
    nIterationsList.foreach{ nIterations =>
      frequencyThresholdList.foreach{ frequencyThreshold =>
        ambiguityThresholdList.foreach { ambiguityThreshold =>
          val parameters = s"nIterations = $nIterations frequencyThresholdList = $frequencyThreshold ambiguityThreshold = $ambiguityThreshold"
          println(parameters)
          val metrics = benchMarkWordSegmenter(nIterations, frequencyThreshold, ambiguityThreshold)
          println(s"Precision = ${metrics._1}  Recall = ${metrics._2}  FScore = ${metrics._3}")
          val accuracyMetrics = AccuracyByParameter(nIterations, frequencyThreshold, ambiguityThreshold,
            metrics._1, metrics._2, metrics._3)
          accuracyByParameters = accuracyMetrics :: accuracyByParameters
        }
      }
    }
    exportMetrics(accuracyByParameters)
  }

  "WordSegmenterBenchmark with parameters" should "output metrics" ignore {
    val nIterations = 7
    val frequencyThreshold = 30
    val ambiguityThreshold = 0.99
    val parameters = s"nIterations = $nIterations frequencyThresholdList = $frequencyThreshold ambiguityThreshold = $ambiguityThreshold"
    println(parameters)
    val metrics = benchMarkWordSegmenter(nIterations, frequencyThreshold, ambiguityThreshold)
    println(s"Precision = ${metrics._1}  Recall = ${metrics._2}  FScore = ${metrics._3}")
  }

  it should "benchmark a pipeline pretrained model" ignore {
    val tokenizerPipeline = PipelineModel.load("./pipeline_wordsegmenter_model")
    val metrics = evaluateModel(tokenizerPipeline)
    println(s"Precision = ${metrics._1}  Recall = ${metrics._2}  FScore = ${metrics._3}")
  }

  it should "benchmark a word segmenter pretrained model" ignore {
    val emptyDataset = PipelineModels.dummyDataset
    val modelPath = "./tmp_ontonotes_poc_dev_model"
    val wordSegmenter = WordSegmenterModel.load(modelPath)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, wordSegmenter))
    val tokenizerPipeline = pipeline.fit(emptyDataset)
    val metrics = evaluateModel(tokenizerPipeline)
    println(s"Precision = ${metrics._1}  Recall = ${metrics._2}  FScore = ${metrics._3}")
  }

  private def benchMarkWordSegmenter(nIterations: Int, frequencyThreshold: Int, ambiguityThreshold: Double,
                                     exportMetricsBySentence: Boolean = false):
    (Double, Double, Double) = {
    val trainingDataSet = POS().readDataset(ResourceHelper.spark, trainingDataSetFile)

    val wordSegmenter = new WordSegmenterApproach()
      .setInputCols("document")
      .setOutputCol("token")
      .setPosColumn("tags")
      .setNIterations(nIterations)
      .setFrequencyThreshold(frequencyThreshold)
      .setAmbiguityThreshold(ambiguityThreshold)
    println("Training Word Segmenter Model...")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, wordSegmenter))
    val pipelineModel = pipeline.fit(trainingDataSet)

    evaluateModel(pipelineModel, exportMetricsBySentence)
  }

  private def evaluateModel(pipelineModel: PipelineModel, exportMetricsBySentence: Boolean = false): (Double, Double, Double) = {
    val testingDataSet = ResourceHelper.spark.read.text(testingDataSetFile)
      .withColumnRenamed("value", "text")
    val tokenizerDataSet = pipelineModel.transform(testingDataSet)
    tokenizerDataSet.select("token.result").show(1, false)

    val predictedTokensBySentences = tokenizerDataSet.select("token.result").rdd.map{ row=>
      val resultSeq: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]]
      resultSeq.toList
    }.collect().toList
      .filter(predictedTokens => predictedTokens.nonEmpty)

    val realTokensBySentence = getGroundTruthTokens.filter(realTokens => realTokens.nonEmpty)
    val metricsBySentences = predictedTokensBySentences.zipWithIndex.map{ case(predictedTokens, index) =>
      val realTokens = realTokensBySentence(index)
      computeMetrics(index, predictedTokens, realTokens)
    }

    if (exportMetricsBySentence) {
      exportMetricsBySentenceFile(metricsBySentences)
    }

    val averagePrecision = metricsBySentences.map(_.precision).sum / metricsBySentences.length
    val averageRecall = metricsBySentences.map(_.recall).sum / metricsBySentences.length
    val averageFScore = metricsBySentences.map(_.fScore).sum / metricsBySentences.length
    (averagePrecision, averageRecall, averageFScore)
  }

  private def getGroundTruthTokens: List[List[String]] = {
    val externalResource = ExternalResource(groundTruthDataSetFile, ReadAs.TEXT, Map("format" -> "text"))
    val groundTruthTokens = ResourceHelper.parseLines(externalResource)
      .map(groundTruth => groundTruth.split(" ").filter(token => token != "").toList).toList
    groundTruthTokens
  }

  private def computeMetrics(index: Int, predictedTokens: List[String], realTokens: List[String]): Metrics = {
    val realPositives = realTokens.length.toDouble
    val truePositives = realTokens.map(realToken => predictedTokens.contains(realToken))
      .count(positive => positive).toDouble
    val falsePositive = predictedTokens.length.toDouble - truePositives

    val recall = truePositives / realPositives
    val precision = truePositives / (truePositives + falsePositive)
    val fScore = if ((precision + recall) == 0.0) 0.0 else 2.0 * ((precision * recall) / (precision + recall))
    Metrics(index, precision, recall, fScore)
  }

  private def exportMetricsBySentenceFile(metricsBySentences: List[Metrics]): Unit = {
    import ResourceHelper.spark.implicits._

    Encoders.product[Metrics].schema
    val metricsDataFrame = metricsBySentences.toDS().toDF()
    metricsDataFrame.show(5, false)
    exportFile(metricsDataFrame, "word_segmenter_metrics_by_sentence")
  }

  private def exportMetrics(metrics: List[AccuracyByParameter]): Unit = {
    import ResourceHelper.spark.implicits._

    Encoders.product[AccuracyByParameter].schema
    val metricsDataFrame = metrics.toDS().toDF()
    metricsDataFrame.show(5, false)
    exportFile(metricsDataFrame, "word_segmenter_metrics")
  }

  private def exportFile(dataFrame: DataFrame, fileName: String): Unit = {
    dataFrame.coalesce(1)
      .write.format("csv")
      .option("header", "true")
      .save(fileName)
  }

}
