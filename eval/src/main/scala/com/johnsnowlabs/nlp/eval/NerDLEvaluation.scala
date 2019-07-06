package com.johnsnowlabs.nlp.eval

import java.io.File

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.util.{Benchmark, PipelineModels}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable

object NerDLEvaluation extends App {

  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "8G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  import spark.implicits._

  spark.sparkContext.setLogLevel("OFF")

  println("Accuracy Metrics for NER DL")

  private val trainFile = "./eng.train"
  private val testFiles = "./eng.testa"
  private val numberOfEpochs = 1
  private val emptyDataSet = PipelineModels.dummyDataset
  evaluateDataSet(testFiles, "IOB", "model_path", "trainFile_path")

  def apply(testFile: String, format: String, modelPath: String, trainFile: String): Unit = {
    evaluateDataSet(testFile, format, modelPath, trainFile)
  }

  private def evaluateDataSet(testFiles: String, format: String="",
                              modelPath: String, trainFile: String): Unit = {
    val nerDataSet = CoNLL().readDataset(spark, testFiles).cache()
    val labels = getEntitiesLabels(nerDataSet, "label.result", format)
    println("Entities: " + labels)
    val predictionDataSet = getPredictionDataSet(modelPath, trainFile, nerDataSet)
    val evaluationDataSet = getEvaluationDataSet(predictionDataSet, labels, format)
    println("Evaluation Dataset")
    evaluationDataSet.show(5, false)
    computeAccuracy(evaluationDataSet, labels)
  }

  def getPredictionDataSet(modelPath: String, trainFile: String, nerDataSet: Dataset[_]): Dataset[_] = {
    val nerModel = getNerModel(modelPath, trainFile)
    var predictionDataSet: Dataset[_] = PipelineModels.dummyDataset
    Benchmark.measure("Time to transform") {
      predictionDataSet = nerModel.transform(nerDataSet)
        .select(col("label.result").alias("label"),
          col("ner.result").alias("prediction"))
    }
    Benchmark.measure("Time to show prediction dataset") {
      predictionDataSet.show(5)
    }
    predictionDataSet
  }

  def getEntitiesLabels(dataSet: Dataset[_], column: String, format: String): List[String] = {
    val entities: Seq[String] = dataSet.select(dataSet(column)).collect.flatMap(_.toSeq).flatMap { entitiesArray =>
      val entities = entitiesArray.asInstanceOf[mutable.WrappedArray[String]]
      if (format == "IOB") {
        entities
      } else {
        entities.map(element => element.replace("I-", "").replace("B-", ""))
      }
    }
    entities.toList.distinct
  }

  def getNerModel(modelPath: String, trainFile: String): PipelineModel = {
    if (new File(modelPath).exists()) {
      PipelineModel.load(modelPath)
    } else {
      var model: PipelineModel = null
      Benchmark.time("Time to train") {
        val nerPipeline = getNerPipeline(trainFile)
        model = nerPipeline.fit(emptyDataSet)
      }
      model.write.overwrite().save(modelPath)
      model
    }
  }

  def getNerPipeline(trainFile: String): Pipeline = {

    val trainDataSet = CoNLL().readDataset(spark, trainFile)
    println("Train Dataset")
    trainDataSet.show(5)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val glove = new WordEmbeddings()
      .setInputCols("sentence", "token")
      .setOutputCol("glove")
      .setEmbeddingsSource("./glove.6B.100d.txt",
        100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

    val readyData = glove.fit(trainDataSet).transform(trainDataSet).cache()

    val nerTagger = new NerDLApproach()
      .setInputCols(Array("sentence", "token", "glove"))
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(10)
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
        glove,
        nerTagger,
        converter))

    pipeline

  }

  def getEvaluationDataSet(dataSet: Dataset[_], labels: List[String], format: String): Dataset[_] = {
    val labelAndPrediction: Seq[(String, String)] = dataSet.select("label", "prediction").rdd.map { row =>
      val labelColumn: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]]
      val predictionColumn: Seq[String] = row.get(1).asInstanceOf[mutable.WrappedArray[String]]
      if (format == "IOB") {
        (labelColumn.toList, predictionColumn.toList)
      } else {
        val groupLabelColumn = labelColumn.map(element => element.replace("I-", "")
          .replace("B-", ""))
        val groupPredictionColumn = predictionColumn.map(element => element.replace("I-", "")
          .replace("B-", ""))
        (groupLabelColumn.toList, groupPredictionColumn.toList)
      }
    }.collect().flatMap(row => row._1 zip row._2)
    val evaluationDataSet = labelAndPrediction.toDF("label", "prediction")
    evaluationDataSet
      .withColumn("labelIndex", getLabelIndex(labels)(col("label")))
      .withColumn("predictionIndex", getLabelIndex(labels)(col("prediction")))
      .filter(col("label") =!= 'O')
  }

  private def getLabelIndex(labels: List[String]) = udf { label: String =>
    val index = labels.indexOf(label)
    index.toDouble
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
    predictedLabels.foreach { predictedLabel =>
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
