package com.johnsnowlabs.nlp.eval

import java.io.File

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._

import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.util.{Benchmark, PipelineModels}

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions._

object NerCrfEvaluation extends App {

  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "8G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  import spark.implicits._

  println("Accuracy Metrics for NER CEF")

  private val trainFile = "./eng.train"
  private val testFile = "./eng.testa"
  private val trainDataSet = CoNLL().readDataset(spark, trainFile).cache()
  println("Train Dataset")
  trainDataSet.show(5)
  private val numberOfEpochs = 1
  evaluateDataSet("Testing Dataset", testFile, format = "IOB")
  evaluateDataSet("Testing Dataset", testFile)

  def getNerPipeline: Pipeline = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setTrimAndClearNewLines(false)

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")
      .setPrefixPattern("\\A([^\\s\\p{L}$\\.']*)")
      .setIncludeDefaults(false)

    val glove = new WordEmbeddings()
      .setInputCols("sentence", "token")
      .setOutputCol("glove")
      .setEmbeddingsSource("./glove.6B.100d.txt",
        100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

    val readyData = glove.fit(trainDataSet).transform(trainDataSet).cache()

    val posTagger = PerceptronModel.pretrained()

    val nerTagger = new NerCrfApproach()
      .setInputCols(Array("sentence", "token", "pos", "glove"))
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(numberOfEpochs)
      .setRandomSeed(0)
      .setVerbose(2)
      .fit(readyData)

    val converter = new NerConverter()
      .setInputCols(Array("document", "token", "ner"))
      .setOutputCol("ner_span")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        glove,
        nerTagger,
        converter))

    pipeline

  }

  def getTokenPipeline: Pipeline = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setTrimAndClearNewLines(false)

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")
      .setPrefixPattern("\\A([^\\s\\p{L}$\\.']*)")
      .setIncludeDefaults(false)

    val pipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        sentenceDetector,
        tokenizer))

    pipeline

  }

  def getNerModel(trainDataSet: Dataset[_]): PipelineModel = {
    if (new File("./ner_crf_model_"+numberOfEpochs).exists()) {
      PipelineModel.load("./ner_crf_model_"+numberOfEpochs)
    } else {
      var model: PipelineModel = null
      Benchmark.time("Time to train") {
        val nerPipeline = getNerPipeline
        model = nerPipeline.fit(trainDataSet)
      }
      model.write.overwrite().save("./ner_crf_model_"+numberOfEpochs)
      model
    }
  }

  def getTrainDataSetWithTokens(pathDataSet: String): Dataset[_] = {
    var trainDataSet = spark.read.option("delimiter", " ").csv(pathDataSet)
      .withColumnRenamed("_c0", "text")
      .withColumnRenamed("_c3", "ground_truth")

    trainDataSet = trainDataSet.select("text", "ground_truth")
    trainDataSet = trainDataSet.filter("text != '-DOCSTART-'")
    val pipeline = getTokenPipeline
    pipeline.fit(trainDataSet).transform(trainDataSet)
      .filter("ground_truth is not null")
  }

  def getEntitiesLabels(dataSet: Dataset[_], column: String, format: String): List[String] = {
    val labels = dataSet.select(dataSet(column)).distinct()
      .rdd.map(row => row.get(0)).collect().toList
      .filter(_ != null)
    if (format == "IOB"){
      labels.asInstanceOf[List[String]]
    } else {
      labels.asInstanceOf[List[String]]
        .map(element => element.replace("I-", "").replace("B-", ""))
        .distinct
    }
  }

  def getEvaluationDataSet(dataSet: Dataset[_], labels: List[String], format: String): Dataset[_] = {

    dataSet
      .withColumnRenamed("result", "prediction")
      .withColumnRenamed("ground_truth", "label")
      .withColumn("prediction", col("prediction").cast("string"))
      .withColumn("prediction", cleanPrediction(col("prediction")))
      .withColumn("label", formatEntities(format)(col("label")))
      .withColumn("prediction", formatEntities(format)(col("prediction")))
      .withColumn("labelIndex", getLabelIndex(labels)(col("label")))
      .withColumn("predictionIndex", getLabelIndex(labels)(col("prediction")))
      .filter(col("label") =!= 'O')
  }

  private def getLabelIndex(labels: List[String]) = udf { label: String =>
    val index = labels.indexOf(label)
    index.toDouble
  }

  private def cleanPrediction = udf { label: String =>
    label.replace("[", "").replace("]", "")
  }

  private def formatEntities(format: String) = udf { entity: String =>
    if (format == "IOB") {
      entity
    } else {
      entity.replace("I-", "").replace("B-", "")
    }
  }

  private def evaluateDataSet(dataSetType: String, dataSetFile: String, format: String = ""): Unit = {
    val trainDataSet = getTrainDataSetWithTokens(dataSetFile)
    val labels = getEntitiesLabels(trainDataSet, "ground_truth", format)
    println("Entities: " + labels)
    val nerModel = getNerModel(trainDataSet)
    var predictionDataSet: Dataset[_] = PipelineModels.dummyDataset
    println(s"Accuracy for $dataSetType")
    Benchmark.measure("Time to transform") {
      predictionDataSet = nerModel.transform(trainDataSet)
        .select("ground_truth", "ner.result")
    }
    Benchmark.measure("Time to show prediction dataset") {
      predictionDataSet.show(5)
    }
    val evaluationDataSet = getEvaluationDataSet(predictionDataSet, labels, format = format)
    evaluationDataSet.show(5, false)
    computeAccuracy(evaluationDataSet, labels)
  }

  private def computeAccuracy(dataSet: Dataset[_], labels: List[String]): Unit = {
    val predictionLabelsRDD = dataSet.select("predictionIndex", "labelIndex")
      .map(r => (r.getDouble(0), r.getDouble(1)))
    val metrics = new MulticlassMetrics(predictionLabelsRDD.rdd)
    val accuracy = (metrics.accuracy * 1000).round / 1000.toDouble
    println(s"Accuracy = $accuracy")
    computeAccuracyByEntity(metrics, labels)
    computeMicroAverage(metrics)
  }

  private def computeAccuracyByEntity(metrics: MulticlassMetrics, labels: List[String]): Unit = {
    val predictedLabels = metrics.labels
    predictedLabels.foreach{predictedLabel =>
      val entity = labels(predictedLabel.toInt)
      val precision = (metrics.precision(predictedLabel) * 1000).round / 1000.toDouble
      val recall = (metrics.recall(predictedLabel) * 1000).round / 1000.toDouble
      val f1Score = (metrics.fMeasure(predictedLabel) * 1000).round / 1000.toDouble
      println(s"$entity: Precision = $precision, Recall = $recall, F1-Score = $f1Score")
    }
  }

  def computeMicroAverage(metrics: MulticlassMetrics): Unit = {
    var totalP = 0.0
    var totalR = 0.0
    var totalClassNum = 0

    val labels = metrics.labels

    labels.foreach { l =>
      totalClassNum = totalClassNum + 1
      totalP = totalP + metrics.precision(l)
      totalR = totalR + metrics.recall(l)
    }
    totalP = totalP/totalClassNum
    totalR = totalR/totalClassNum
    val microAverage = 2 * ((totalP*totalR) / (totalP+totalR))
    println(s"Micro-average F-1 Score:  ${(microAverage * 1000).round / 1000.toDouble}")
  }

}