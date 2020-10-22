package com.johnsnowlabs.benchmarks.tokenizer

import java.io.{BufferedWriter, File, FileWriter}

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.{WordSegmenterApproach, WordSegmenterModel}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Encoders
import org.scalatest.FlatSpec

import scala.collection.mutable

class WordSegmenterBenchmark extends FlatSpec {

  private val trainingDataSetFile = "train.ws.utf8"
  private val testingDataSetFile = "test.untokenized.utf8"
  private val groundTruthDataSetFile = "groundtruth.dev.ws.utf8"

  private val minFrequencyList = List(2.05E-5, 1.05E-5, 5.05E-6, 4.05E-6, 3.05E-6, 2.05E-6, 1.05E-6)
  private val minAggregationList = List(1, 2, 3, 4, 5, 6, 7)
  private val maxWordLengthList = List(2, 3, 4, 5)
  private val minEntropyList = List(0.1, 0.2, 0.3, 0.4, 0.5)

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val sentence = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

  "WordSegmenterBenchmark with a set of parameters" should "output metrics" ignore {
    val file = new File("word_segmenter_metrics.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    minFrequencyList.foreach{ minFrequency =>
      minAggregationList.foreach{ minAggregation =>
        maxWordLengthList.foreach { maxWordLength =>
          minEntropyList.foreach{ minEntropy =>
            val parameters = s"MinFrequency = $minFrequency  MaxWordLength = $maxWordLength  MinAggregation = $minAggregation MinEntropy = $minEntropy"
            println(parameters)
            val metrics = benchMarkWordSegmenter(minFrequency, maxWordLength, minAggregation, minEntropy)
            println(s"Precision = ${metrics._1}  Recall = ${metrics._2}  FScore = ${metrics._3}")
            bw.write(parameters + "|" + metrics._1 + "|" + metrics._2 + "|" + metrics._3 + "\n")
          }
        }
      }
    }
    bw.close()
  }

  "WordSegmenterBenchmark with parameters" should "output metrics" ignore {
    val minFrequency = 5.05E-6
    val maxWordLength = 4
    val minEntropy = 0.1
    val minAggregation = 1
    val parameters = s"MinFrequency = $minFrequency  MaxWordLength = $maxWordLength  MinAggregation = $minAggregation MinEntropy = $minEntropy"
    println(parameters)
    val metrics = benchMarkWordSegmenter(minFrequency, maxWordLength, minAggregation, minEntropy)
    println(s"Precision = ${metrics._1}  Recall = ${metrics._2}  FScore = ${metrics._3}")
  }

  it should "benchmark a pipeline pretrained model" ignore {
    val tokenizerPipeline = PipelineModel.load("./pipeline_wordsegmenter_model")
    val metrics = evaluateModel(tokenizerPipeline)
    println(s"Precision = ${metrics._1}  Recall = ${metrics._2}  FScore = ${metrics._3}")
  }

  it should "benchmark a word segmenter pretrained model" ignore {
    val emptyDataset = PipelineModels.dummyDataset
    val modelPath = "./word_segmenter_model"
    val wordSegmenter = WordSegmenterModel.load(modelPath)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, wordSegmenter))
    val tokenizerPipeline = pipeline.fit(emptyDataset)
    val metrics = evaluateModel(tokenizerPipeline)
    println(s"Precision = ${metrics._1}  Recall = ${metrics._2}  FScore = ${metrics._3}")
  }

  private def benchMarkWordSegmenter(minFrequency: Double, maxWordLength: Int, minAggregation: Double,
                                     minEntropy: Double): (Double, Double, Double) = {
    val trainingDataSet = ResourceHelper.spark.read.text(trainingDataSetFile)
      .withColumnRenamed("value", "text")

    val wordSegmenter = new WordSegmenterApproach()
    .setInputCols("sentence")
    .setOutputCol("token")
    .setMinFrequency(minFrequency)
    .setMaxWordLength(maxWordLength)
    .setMinAggregation(minAggregation)
    .setMinEntropy(minEntropy)
    .setWordSegmentMethod("ALL")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, wordSegmenter))
    val tokenizerPipeline = pipeline.fit(trainingDataSet)

    evaluateModel(tokenizerPipeline)
  }

  private def evaluateModel(pipelineModel: PipelineModel): (Double, Double, Double) = {
    val testingDataSet = ResourceHelper.spark.read.text(testingDataSetFile)
      .withColumnRenamed("value", "text")
    val tokenizerDataSet = pipelineModel.transform(testingDataSet)

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

    exportMetrics(metricsBySentences)

    val averagePrecision = metricsBySentences.map(_.precision).sum / metricsBySentences.length
    val averageRecall = metricsBySentences.map(_.recall).sum / metricsBySentences.length
    val averageFScore = metricsBySentences.map(_.fScore).sum / metricsBySentences.length
    (averagePrecision, averageRecall, averageFScore)
  }

  private def exportMetrics(metricsBySentences: List[Metrics]): Unit = {
    import ResourceHelper.spark.implicits._

    Encoders.product[Metrics].schema
    val metricsDataSet = metricsBySentences.toDS().toDF()
    metricsDataSet.show()

    metricsDataSet.coalesce(1)
      .write.format("csv")
      .option("header", "true")
      .save("word_segmenter_metrics")
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

}
