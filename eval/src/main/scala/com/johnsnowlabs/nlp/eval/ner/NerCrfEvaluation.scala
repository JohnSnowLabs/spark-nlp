package com.johnsnowlabs.nlp.eval.ner

import com.johnsnowlabs.nlp.annotator.{NerCrfApproach, NerCrfModel, WordEmbeddings, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.eval.util.{GoldTokenizer, LoggingData}
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable

class NerCrfEvaluation(sparkSession: SparkSession, testFile: String, tagLevel: String = "") {

  import sparkSession.implicits._

  val goldTokenizer = new GoldTokenizer(sparkSession)
  val loggingData = new LoggingData("LOCAL", this.getClass.getSimpleName, "Named Entity Recognition")

  private case class NerEvalCrfConfiguration(trainFile: String, wordEmbeddings: WordEmbeddings,
                                            nerCrfModel: NerCrfModel, nerCrfApproach: NerCrfApproach)

  def computeAccuracyModel(nerCrfModel: NerCrfModel): Unit = {
    loggingData.logNerCrfParams(nerCrfModel)
    val nerEvalDLConfiguration = NerEvalCrfConfiguration("", null, nerCrfModel, null)
    computeAccuracy(nerEvalDLConfiguration)
    loggingData.closeLog()
  }

  def computeAccuracyAnnotator(trainFile:String, nerCrfApproach: NerCrfApproach, wordEmbeddings: WordEmbeddings): Unit = {
    loggingData.logNerCrfParams(nerCrfApproach)
    val nerEvalDLConfiguration = NerEvalCrfConfiguration(trainFile, wordEmbeddings, null, nerCrfApproach)
    computeAccuracy(nerEvalDLConfiguration)
    loggingData.closeLog()
  }

  private def computeAccuracy(nerEvalCrfConfiguration: NerEvalCrfConfiguration): Unit = {
    import sparkSession.implicits._
    val entityLabels = getEntityLabels(nerEvalCrfConfiguration, tagLevel)
    val evaluationDataSet = getEvaluationDataSet(nerEvalCrfConfiguration, entityLabels, tagLevel)
    val predictionLabelsRDD = evaluationDataSet.select("predictionIndex", "labelIndex")
      .map(r => (r.getDouble(0), r.getDouble(1)))
    val metrics = new MulticlassMetrics(predictionLabelsRDD.rdd)
    NerMetrics.computeAccuracy(metrics, loggingData)
    NerMetrics.computeAccuracyByEntity(metrics, entityLabels, loggingData)
    NerMetrics.computeMicroAverage(metrics, loggingData)
  }

  private def getEntityLabels(nerEvalCrfConfiguration: NerEvalCrfConfiguration, tagLevel: String): List[String] = {
    val joinedDataSet = getJoinedDataSet(nerEvalCrfConfiguration)
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

  private def getJoinedDataSet(nerEvalCrfConfiguration: NerEvalCrfConfiguration): Dataset[_] = {

    val testDataSet = goldTokenizer.getTestTokensTagsDataSet(testFile)
    val predictionDataSet = getPredictionDataSet(nerEvalCrfConfiguration)

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

  private def getPredictionDataSet(nerEvalCrfConfiguration: NerEvalCrfConfiguration): Dataset[_] = {

    val testDataSet = goldTokenizer.getGoldenTokenizer(testFile)
    var predictionDataSet: Dataset[_] = null

    if (nerEvalCrfConfiguration.nerCrfModel == null) {

      val trainDataSet = CoNLL().readDataset(sparkSession, nerEvalCrfConfiguration.trainFile)
      val embeddings = nerEvalCrfConfiguration.wordEmbeddings.fit(trainDataSet)
      val embeddingsTrain = embeddings.transform(trainDataSet)
      val nerModel = nerEvalCrfConfiguration.nerCrfApproach.fit(embeddingsTrain)
      val embeddingsTest = embeddings.transform(testDataSet)

      predictionDataSet = nerModel.transform(embeddingsTest)

    }
    else {
      val embeddings = WordEmbeddingsModel.pretrained()
        .setInputCols("document", "token")
        .setOutputCol("embeddings")
        .transform(testDataSet)

      val nerModel = nerEvalCrfConfiguration.nerCrfModel
        .setInputCols("document", "token", "pos" ,"embeddings")
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

  private def getEvaluationDataSet(nerEvalCrfConfiguration: NerEvalCrfConfiguration, entityLabels: List[String],
                                   tagLevel: String): Dataset[_] = {

    val joinedDataSet = getJoinedDataSet(nerEvalCrfConfiguration)
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
