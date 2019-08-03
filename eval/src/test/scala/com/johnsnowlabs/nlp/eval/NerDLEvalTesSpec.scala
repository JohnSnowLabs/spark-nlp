package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotator.{NerDLApproach, NerDLModel, WordEmbeddings}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.eval.ner.NerDLEvaluation
import org.apache.spark.sql.SparkSession
import org.scalatest.FlatSpec

class NerDLEvalTesSpec extends FlatSpec {

  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "8G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  "A NER DL Evaluation with IOB tags" should "display accuracy results for pretrained model" in {

    val testFile = "./eng.testb"
    val nerModel = NerDLModel.pretrained()
    val tagLevel = "IOB"

    val nerDLEvaluation = new NerDLEvaluation(spark, testFile, tagLevel)
    nerDLEvaluation.computeAccuracyModel(nerModel)

  }

  "A NER DL Evaluation with IOB tags" should "display accuracy results" in {
    val trainFile = "./eng.train"
    val testFile = "./eng.testb"
    val tagLevel = "IOB"

    val embeddings = new WordEmbeddings()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setEmbeddingsSource("./glove.6B.100d.txt",
        100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

    val nerApproach = new NerDLApproach()
      .setInputCols(Array("document", "token", "embeddings"))
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(10)
      .setRandomSeed(0)
      .setVerbose(2)

    val nerDLEvaluation = new NerDLEvaluation(spark, testFile, tagLevel)
    nerDLEvaluation.computeAccuracyAnnotator(trainFile, nerApproach, embeddings)

  }

  "A NER DL Evaluation" should "display accuracy results for pretrained model" in {

    val testFile = "./eng.testb"
    val nerModel = NerDLModel.pretrained()

    val nerDLEvaluation = new NerDLEvaluation(spark, testFile)
    nerDLEvaluation.computeAccuracyModel(nerModel)

  }

  "A NER DL Evaluation" should "display accuracy results" in {
    val trainFile = "./eng.train"
    val testFile = "./eng.testb"

    val embeddings = new WordEmbeddings()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setEmbeddingsSource("./glove.6B.100d.txt",
        100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

    val nerApproach = new NerDLApproach()
      .setInputCols(Array("document", "token", "embeddings"))
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(10)
      .setRandomSeed(0)
      .setVerbose(2)

    val nerDLEvaluation = new NerDLEvaluation(spark, testFile)
    nerDLEvaluation.computeAccuracyAnnotator(trainFile, nerApproach, embeddings)

  }

}
