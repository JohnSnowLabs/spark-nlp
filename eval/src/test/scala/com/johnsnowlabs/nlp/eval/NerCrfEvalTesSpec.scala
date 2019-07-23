package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotator.{NerCrfApproach, NerCrfModel, WordEmbeddings}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.eval.ner.NerCrfEvaluation
import org.apache.spark.sql.SparkSession
import org.scalatest.FlatSpec

class NerCrfEvalTesSpec extends FlatSpec {

  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "8G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  "A NER CRF Evaluation with IOB tags" should "display accuracy results" in {

    val testFile = "./eng.testb"
    val nerModel = NerCrfModel.pretrained()
    val tagLevel = "IOB"

    val nerCrfEvaluationGoldToken = new NerCrfEvaluation(spark, testFile, tagLevel)
    nerCrfEvaluationGoldToken.computeAccuracyModel(nerModel)

  }

  "A NER CRF Evaluation with IOB tags" should "display accuracy results for pretrained model" in {
    val trainFile = "./eng.train"
    val testFile = "./eng.testb"
    val tagLevel = "IOB"

    val embeddings = new WordEmbeddings()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setEmbeddingsSource("./glove.6B.100d.txt",
        100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

    val nerApproach = new NerCrfApproach()
      .setInputCols(Array("document", "token", "pos", "embeddings"))
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(10)
      .setRandomSeed(0)
      .setVerbose(2)

    val nerCrfEvaluation = new NerCrfEvaluation(spark, testFile, tagLevel)
    nerCrfEvaluation.computeAccuracyAnnotator(trainFile, nerApproach, embeddings)

  }

  "A NER CRF Evaluation" should "display accuracy results" in {

    val testFile = "./eng.testb"
    val nerModel = NerCrfModel.pretrained()

    val nerCrfEvaluationGoldToken = new NerCrfEvaluation(spark, testFile)
    nerCrfEvaluationGoldToken.computeAccuracyModel(nerModel)

  }

  "A NER CRF Evaluation" should "display accuracy results for pretrained model" in {
    val trainFile = "./eng.train"
    val testFile = "./eng.testb"

    val embeddings = new WordEmbeddings()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setEmbeddingsSource("./glove.6B.100d.txt",
        100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

    val nerApproach = new NerCrfApproach()
      .setInputCols(Array("document", "token", "pos", "embeddings"))
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(10)
      .setRandomSeed(0)
      .setVerbose(2)

    val nerCrfEvaluation = new NerCrfEvaluation(spark, testFile)
    nerCrfEvaluation.computeAccuracyAnnotator(trainFile, nerApproach, embeddings)

  }

}
