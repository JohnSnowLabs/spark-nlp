package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec

class OpenAIEmbeddingsTest extends AnyFlatSpec {

  private val spark = SparkSession
    .builder()
    .appName("test")
    .master("local[*]")
    .config("spark.driver.memory", "16G")
    .config("spark.driver.maxResultSize", "0")
    .config("spark.kryoserializer.buffer.max", "2000M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config(
      "spark.jsl.settings.openai.api.key",
      "" // Set your OpenAI API key here...
    )
    .getOrCreate()

  import spark.implicits._
  private val documentAssembler =
    new com.johnsnowlabs.nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")

  "OpenAIEmbeddings" should "generate a completion for prompts" taggedAs SlowTest in {
    // Set OPENAI_API_KEY env variable to make this work
    val promptDF = Seq("The food was delicious and the waiter...").toDS.toDF("text")

    promptDF.show(false)

    val openAIEmbeddings = new OpenAIEmbeddings()
      .setInputCols("document")
      .setOutputCol("embeddings")
      .setModel("text-embedding-ada-002")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, openAIEmbeddings))
    val completionDF = pipeline.fit(promptDF).transform(promptDF)
    completionDF.select("embeddings").show(false)
  }

  "OpenAIEmbeddings" should "work with escape chars" taggedAs SlowTest in {
    val data = Seq(
      (1, "Hello \"World\""),
      (2, "Hello \n World"),
      (3, "Hello \t World"),
      (4, "Hello \r World"),
      (5, "Hello \b World"),
      (6, "Hello \f World"),
      (7, "Hello \\ World"))
    val columns = Seq("id", "text")
    val testDF = spark.createDataFrame(data).toDF(columns: _*)

    val openAIEmbeddings = new OpenAIEmbeddings()
      .setInputCols("document")
      .setOutputCol("embeddings")
      .setModel("text-embedding-ada-002")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, openAIEmbeddings))
    val resultDF = pipeline.fit(testDF).transform(testDF)
    resultDF.select("embeddings").show(false)
  }

}
