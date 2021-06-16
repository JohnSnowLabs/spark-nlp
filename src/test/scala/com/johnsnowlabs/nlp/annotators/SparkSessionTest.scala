package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, Suite}

trait SparkSessionTest extends BeforeAndAfterAll { this: Suite =>

  val spark: SparkSession = SparkAccessor.spark
  val sparkContext: SparkContext = spark.sparkContext
  val documentAssembler = new DocumentAssembler()
  val tokenizer = new Tokenizer()
  val tokenizerPipeline = new Pipeline()

  override def beforeAll(): Unit = {
    super.beforeAll()

    documentAssembler
      .setInputCol("text")
      .setOutputCol("document")

    tokenizer
      .setInputCols("document")
      .setOutputCol("token")

    tokenizerPipeline.setStages(Array(documentAssembler, tokenizer))

  }

  override def afterAll(): Unit = {
    try super.afterAll()
    spark.stop()
  }

}
