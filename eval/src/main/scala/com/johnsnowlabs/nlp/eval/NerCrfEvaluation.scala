package com.johnsnowlabs.nlp.eval

import java.io.File

import com.johnsnowlabs.nlp.annotator.{NerConverter, NerCrfApproach, PerceptronModel, SentenceDetector, WordEmbeddings}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.eval.util.LoggingData
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.util.{Benchmark, PipelineModels}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

import scala.collection.mutable

class NerCrfEvaluation(testFile: String, format: String) {

  var loggingData: LoggingData = _

  private case class NerEvalCrfConfiguration(trainFile: String, format: String, modelPath: String,
                                             sparkSession: SparkSession, NerCrfApproach: NerCrfApproach,
                                             wordEmbeddings: WordEmbeddings)

  def computeAccuracyAnnotator(modelPath: String, trainFile: String, NerCrfApproach: NerCrfApproach,
                               wordEmbeddings: WordEmbeddings): Unit = {

    val spark = SparkSession.builder()
      .appName("benchmark")
      .master("local[*]")
      .config("spark.driver.memory", "8G")
      .config("spark.kryoserializer.buffer.max", "200M")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()

    val nerEvalCrfConfiguration = NerEvalCrfConfiguration(trainFile, format, modelPath, spark,
      NerCrfApproach, wordEmbeddings)

    loggingData = new LoggingData("LOCAL", this.getClass.getSimpleName, "Named Entity Recognition")
    //loggingData.logNerCrfParams(NerCrfApproach)
    evaluateDataSet(testFile, nerEvalCrfConfiguration)
    loggingData.closeLog()
  }

  private def evaluateDataSet(testFile: String, nerEvalCrfConfiguration: NerEvalCrfConfiguration):
  Unit = {
    val nerDataSet = CoNLL().readDataset(nerEvalCrfConfiguration.sparkSession, testFile).cache()
    val labels = getEntitiesLabels(nerDataSet, "label.result", nerEvalCrfConfiguration.format)
    println("Entities: " + labels)
    val predictionDataSet = getPredictionDataSet(nerDataSet, nerEvalCrfConfiguration)
    val evaluationDataSet = getEvaluationDataSet(predictionDataSet, labels, nerEvalCrfConfiguration.format,
      nerEvalCrfConfiguration.sparkSession)
    println("Evaluation Dataset")
    evaluationDataSet.show(5, false)
    computeAccuracy(evaluationDataSet, labels, nerEvalCrfConfiguration.sparkSession)
  }

  def getPredictionDataSet(nerDataSet: Dataset[_], nerEvalCrfConfiguration: NerEvalCrfConfiguration):
  Dataset[_] = {
    val nerModel = getNerModel(nerEvalCrfConfiguration)
    val predictionDataSet = nerModel.transform(nerDataSet)
      .select(col("label.result").alias("label"),
        col("ner.result").alias("prediction"))
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

  def getNerModel(nerEvalCrfConfiguration: NerEvalCrfConfiguration): PipelineModel = {
    if (new File(nerEvalCrfConfiguration.modelPath).exists()) {
      PipelineModel.load(nerEvalCrfConfiguration.modelPath)
    } else {
      var model: PipelineModel = null
      Benchmark.setPrint(false)
      val time = Benchmark.measure(1, false, "Time to train") {
        val nerPipeline = getNerPipeline(nerEvalCrfConfiguration)
        model = nerPipeline.fit(PipelineModels.dummyDataset)
      }
      loggingData.logMetric("training time/s", time)
      model.write.overwrite().save(nerEvalCrfConfiguration.modelPath)
      model
    }
  }

  def getNerPipeline(nerEvalCrfConfiguration: NerEvalCrfConfiguration): Pipeline = {

    val trainDataSet = CoNLL().readDataset(nerEvalCrfConfiguration.sparkSession, nerEvalCrfConfiguration.trainFile)
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

    val posTagger = PerceptronModel.pretrained()

    val readyData = nerEvalCrfConfiguration.wordEmbeddings.fit(trainDataSet).transform(trainDataSet).cache()

    val nerTagger = nerEvalCrfConfiguration.NerCrfApproach.fit(readyData)

    val converter = new NerConverter()
      .setInputCols(Array("document", "token", "ner"))
      .setOutputCol("ner_span")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        nerEvalCrfConfiguration.wordEmbeddings,
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
    loggingData.logMetric("accuracy", accuracy)
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
      loggingData.logMetric(entity + " precision", precision)
      loggingData.logMetric(entity + " recall", recall)
      loggingData.logMetric(entity + " f1-score", f1Score)
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
    loggingData.logMetric("micro-average f1-score", (microAverage * 1000).round / 1000.toDouble)
  }

}
