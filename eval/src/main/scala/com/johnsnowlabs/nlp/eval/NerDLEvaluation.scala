package com.johnsnowlabs.nlp.eval

import java.io.File

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.util.{Benchmark, PipelineModels}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable

object NerDLEvaluation extends App {

  println("Accuracy Metrics for NER DL")
  private case class NerEvalDLConfiguration(trainFile: String, format:String, modelPath: String,
                                            sparkSession: SparkSession, nerDLApproach: NerDLApproach,
                                            wordEmbeddings: WordEmbeddings)

  def apply(testFile: String, format: String, modelPath: String, trainFile: String, nerDLApproach: NerDLApproach,
            wordEmbeddings: WordEmbeddings): Unit = {

    val spark = SparkSession.builder()
      .appName("benchmark")
      .master("local[*]")
      .config("spark.driver.memory", "8G")
      .config("spark.kryoserializer.buffer.max", "200M")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()

    val nerEvalDLConfiguration = NerEvalDLConfiguration(trainFile, format, modelPath, spark,
                                                        nerDLApproach, wordEmbeddings)

    evaluateDataSet(testFile, nerEvalDLConfiguration)
  }

  private def evaluateDataSet(testFile: String, nerEvalDLConfiguration: NerEvalDLConfiguration):
  Unit = {
    val nerDataSet = CoNLL().readDataset(nerEvalDLConfiguration.sparkSession, testFile).cache()
    val labels = getEntitiesLabels(nerDataSet, "label.result", nerEvalDLConfiguration.format)
    println("Entities: " + labels)
    val predictionDataSet = getPredictionDataSet(nerDataSet, nerEvalDLConfiguration)
    val evaluationDataSet = getEvaluationDataSet(predictionDataSet, labels, nerEvalDLConfiguration.format,
      nerEvalDLConfiguration.sparkSession)
    println("Evaluation Dataset")
    evaluationDataSet.show(5, false)
    computeAccuracy(evaluationDataSet, labels, nerEvalDLConfiguration.sparkSession)
  }

  def getPredictionDataSet(nerDataSet: Dataset[_], nerEvalDLConfiguration: NerEvalDLConfiguration):
  Dataset[_] = {
    val nerModel = getNerModel(nerEvalDLConfiguration)
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

  def getNerModel(nerEvalDLConfiguration: NerEvalDLConfiguration): PipelineModel = {
    if (new File(nerEvalDLConfiguration.modelPath).exists()) {
      PipelineModel.load(nerEvalDLConfiguration.modelPath)
    } else {
      var model: PipelineModel = null
      Benchmark.time("Time to train") {
        val nerPipeline = getNerPipeline(nerEvalDLConfiguration)
        model = nerPipeline.fit(PipelineModels.dummyDataset)
      }
      model.write.overwrite().save(nerEvalDLConfiguration.modelPath)
      model
    }
  }

  def getNerPipeline(nerEvalDLConfiguration: NerEvalDLConfiguration): Pipeline = {

    val trainDataSet = CoNLL().readDataset(nerEvalDLConfiguration.sparkSession, nerEvalDLConfiguration.trainFile)
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

    val readyData = nerEvalDLConfiguration.wordEmbeddings.fit(trainDataSet).transform(trainDataSet).cache()

    val nerTagger = nerEvalDLConfiguration.nerDLApproach.fit(readyData)

    val converter = new NerConverter()
      .setInputCols(Array("document", "token", "ner"))
      .setOutputCol("ner_span")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        nerEvalDLConfiguration.wordEmbeddings,
        nerTagger,
        converter))

    pipeline

  }

  def getEvaluationDataSet(dataSet: Dataset[_], labels: List[String], format: String, sparkSession: SparkSession): Dataset[_] = {
   import sparkSession.implicits._
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

  private def computeAccuracy(dataSet: Dataset[_], labels: List[String], sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._
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
