package com.johnsnowlabs.nlp.annotators.classifier

import com.johnsnowlabs.nlp.annotator.{SentenceEmbeddings, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{DocumentRFClassifierApproach, Tokenizer}
import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.scalatest._

class DocumentRFClassifierTestSpec extends FlatSpec {

  "DocumentRFClassifierApproach" should "be trainable in a pipeline" in {
    val spark = ResourceHelper.spark

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

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("sentence", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val docClassifier = new DocumentRFClassifierApproach()

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        embeddings,
        embeddingsSentence,
        docClassifier
      ))

    val pipelineDF = pipeline.fit(trainData).transform(testData)
    pipelineDF.printSchema()
    pipelineDF.show()
    pipelineDF.select("label_output").show(false)

  }
}
