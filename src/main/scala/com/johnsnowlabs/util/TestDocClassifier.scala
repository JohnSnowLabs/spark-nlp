package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{DocClassifierApproach, Tokenizer}
import com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings
import com.johnsnowlabs.nlp.{DocumentAssembler, RecursivePipeline, annotator}
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}

object TestDocClassifier extends App{
  val spark = SparkSession.builder()
    .appName("ER New Session")
    .master("local[16]")
    .config("spark.driver.memory","64G")
    .config("spark.driver.maxResultSize", "0")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryoserializer.buffer.max", "1G")
    .getOrCreate()

  val trainData = spark.createDataFrame(
    spark.sparkContext.parallelize(Seq(
      Row("i want to get a coffee", "buy"),
      Row("i want to buy a coffee", "buy"),
      Row("would you like a coffee", "sell"),
      Row("want me to give a coffee for you", "sell"),
      Row("want to show you the new merchandising", "sell"),
      Row("look at these beautiful shoes for you", "sell"),
      Row("i would kill for a soda", "buy"),
      Row("i would really desire to get a new computer", "buy"),
      Row("i will to buy a laptop", "buy"),
      Row("can i offer you something to drink", "sell")
    )),
    StructType(Seq(StructField("text", StringType), StructField("label", StringType)))
  )

  val testData = spark.createDataFrame(
    spark.sparkContext.parallelize(Seq(
      Row("i want to get a coffee. also offer you an apple", Seq("buy", "sell")),
      Row("would you like a coffee. and maybe sell you something", Seq("sell", "sell")),
      Row("want me to get a coffee for you. or buy one from you", Seq("sell", "buy")),
      Row("look at these beautiful shoes for you. wan to get a laptop", Seq("sell", "buy")),
      Row("i would kill for a soda. and buy some shoes" , Seq("buy", "buy")),
      Row("can i offer you something to drink. give you a meal. and buy the groceries" , Seq("buy", "sell", "buy"))
    )),
    StructType(Seq(StructField("text", StringType), StructField("label", ArrayType(StringType))))
  )

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val sentenceDetector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

  val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  val embeddings = annotator.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols("sentence", "token")
    .setOutputCol("embeddings")

  val sentenceEmbeddings = new SentenceEmbeddings()
    .setInputCols("sentence", "embeddings")
    .setOutputCol("sentence_embeddings")

  val pipeline = new RecursivePipeline()
    .setStages(Array(
      documentAssembler,
      sentenceDetector,
      tokenizer,
      embeddings,
      sentenceEmbeddings
    ))

  val fitPipeline = pipeline.fit(trainData)
  val readyTrainData = fitPipeline.transform(trainData)
  val readyTestData = fitPipeline.transform(testData)

  val docClassifier = new DocClassifierApproach()

  val docClassificationModel = docClassifier.fit(readyTrainData)
  val trainPredictions = docClassificationModel.transform(readyTrainData)

  val preparedTestData = docClassificationModel.prepareData(readyTestData)

  val testPredictions = docClassificationModel.transform(preparedTestData)

  trainPredictions.select("label","label_output").show(10, false)

  testPredictions.select("label", "label_output").show(10, false)
}
