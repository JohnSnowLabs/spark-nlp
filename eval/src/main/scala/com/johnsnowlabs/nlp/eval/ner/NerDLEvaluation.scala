package com.johnsnowlabs.nlp.eval.ner

import com.johnsnowlabs.nlp.annotator.{NerDLApproach, NerDLModel, WordEmbeddings, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.eval.util.{GoldTokenizer, LoggingData}
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable

class NerDLEvaluation(sparkSession: SparkSession, testFile: String, tagLevel: String = "") {

  import sparkSession.implicits._

  val goldTokenizer = new GoldTokenizer(sparkSession)
  val loggingData = new LoggingData("LOCAL", this.getClass.getSimpleName, "Named Entity Recognition")

  private case class NerEvalDLConfiguration(trainFile: String, wordEmbeddings: WordEmbeddings,
                                            nerDLModel: NerDLModel, nerDLApproach: NerDLApproach)

  def computeAccuracyModel(nerDLModel: NerDLModel): Unit = {
    loggingData.logNerDLParams(nerDLModel)
    val nerEvalDLConfiguration = NerEvalDLConfiguration("", null, nerDLModel, null)
    computeAccuracy(nerEvalDLConfiguration)
    loggingData.closeLog()
  }

  def computeAccuracyAnnotator(trainFile:String, nerDLApproach: NerDLApproach, wordEmbeddings: WordEmbeddings): Unit = {
    loggingData.logNerDLParams(nerDLApproach)
    val nerEvalDLConfiguration = NerEvalDLConfiguration(trainFile, wordEmbeddings, null, nerDLApproach)
    computeAccuracy(nerEvalDLConfiguration)
    loggingData.closeLog()
  }

  private def computeAccuracy(nerEvalDLConfiguration: NerEvalDLConfiguration): Unit = {
    import sparkSession.implicits._
    val entityLabels = getEntityLabels(nerEvalDLConfiguration, tagLevel)
    val evaluationDataSet = getEvaluationDataSet(nerEvalDLConfiguration, entityLabels, tagLevel)
    val predictionLabelsRDD = evaluationDataSet.select("predictionIndex", "labelIndex")
      .map(r => (r.getDouble(0), r.getDouble(1)))
    val metrics = new MulticlassMetrics(predictionLabelsRDD.rdd)
    NerMetrics.computeAccuracy(metrics, loggingData)
    NerMetrics.computeAccuracyByEntity(metrics, entityLabels, loggingData)
    NerMetrics.computeMicroAverage(metrics,loggingData)
  }

  private def getEntityLabels(nerEvalDLConfiguration: NerEvalDLConfiguration, tagLevel: String): List[String] = {
    val joinedDataSet = getJoinedDataSet(nerEvalDLConfiguration)
    val entities: Seq[String] = joinedDataSet.select($"testTags").collect.flatMap(_.toSeq).flatMap { entitiesArray =>
      val entities = entitiesArray.asInstanceOf[mutable.WrappedArray[String]]
      if (tagLevel == "IOB") {
        entities
      } else {
        entities.map(element => element.replace("I-", "").replace("B-", ""))
      }

    }

    entities.toList.distinct
  }

  private def getJoinedDataSet(nerEvalDLConfiguration: NerEvalDLConfiguration): Dataset[_] = {

    val testDataSet = goldTokenizer.getTestTokensTagsDataSet(testFile)
    val predictionDataSet = getPredictionDataSet(nerEvalDLConfiguration)

    predictionDataSet
      .join(testDataSet, Seq("id"))
      .withColumn("predictedTokensLength", goldTokenizer.calLengthOfArray($"predictedTokens"))
      .withColumn("predictedTagsLength", goldTokenizer.calLengthOfArray($"predictedTags"))
      .withColumn("testTokensLength", goldTokenizer.calLengthOfArray($"testTokens"))
      .withColumn("testTagsLength", goldTokenizer.calLengthOfArray($"testTags"))
      .withColumn("tokensDiffFromTest", $"testTokensLength" - $"predictedTokensLength")
      .withColumn("tagsDiffFromTest", $"testTagsLength" - $"predictedTagsLength")
      .withColumn("missingTokens", goldTokenizer.extractMissingTokens($"testTokens", $"predictedTokens"))
      .withColumn("missingTags", goldTokenizer.extractMissingTokens($"testTags", $"predictedTags"))
      .withColumn("equalTags", col("predictedTagsLength") === col("testTagsLength"))
  }

  private def getPredictionDataSet(nerEvalDLConfiguration: NerEvalDLConfiguration): Dataset[_] = {

    val testDataSet = goldTokenizer.getGoldenTokenizer(testFile)
    var predictionDataSet: Dataset[_] = null

    if (nerEvalDLConfiguration.nerDLModel == null) {

      val trainDataSet = CoNLL().readDataset(sparkSession, nerEvalDLConfiguration.trainFile)
      val embeddings = nerEvalDLConfiguration.wordEmbeddings.fit(trainDataSet)
      val embeddingsTrain = embeddings.transform(trainDataSet)
      val nerModel = nerEvalDLConfiguration.nerDLApproach.fit(embeddingsTrain)
      val embeddingsTest = embeddings.transform(testDataSet)

      predictionDataSet = nerModel.transform(embeddingsTest)

    }
    else {
      val embeddings = WordEmbeddingsModel.pretrained()
        .setInputCols("document", "token")
        .setOutputCol("embeddings")
        .transform(testDataSet)

      val nerModel = nerEvalDLConfiguration.nerDLModel
        .setInputCols("document", "token", "embeddings")
        .setOutputCol("ner")

      predictionDataSet = nerModel.transform(embeddings)
    }

    predictionDataSet.select(
      $"id",
      $"document",
      $"token",
      $"embeddings",
      $"token.result".alias("predictedTokens"),
      $"ner.result".alias("predictedTags")
    )
  }

  private def getEvaluationDataSet(nerEvalDLConfiguration: NerEvalDLConfiguration, entityLabels: List[String],
                                   tagLevel: String): Dataset[_] = {

    val joinedDataSet = getJoinedDataSet(nerEvalDLConfiguration)
    val labelAndPrediction: Seq[(String, String)] = joinedDataSet.select("testTags", "predictedTags").rdd.map { row =>
      val labelColumn: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]]
      val predictionColumn: Seq[String] = row.get(1).asInstanceOf[mutable.WrappedArray[String]]
      if (tagLevel == "IOB") {
        (labelColumn.toList, predictionColumn.toList)
      } else {
        val groupLabelColumn = labelColumn.map(element => element.replace("I-", "")
          .replace("B-", ""))
        val groupPredictionColumn = predictionColumn.map(element => element.replace("I-", "")
          .replace("B-", ""))
        (groupLabelColumn.toList, groupPredictionColumn.toList)
      }

    }.collect().flatMap(row => row._1 zip row._2)

    labelAndPrediction.toDF("label", "prediction")
      .withColumn("labelIndex", getLabelIndex(entityLabels)(col("label")))
      .withColumn("predictionIndex", getLabelIndex(entityLabels)(col("prediction")))
      .filter(col("label") =!= 'O')
  }

  private def getLabelIndex(labels: List[String]) = udf { label: String =>
    val index = labels.indexOf(label)
    index.toDouble
  }

}
