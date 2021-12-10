package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class GPT2TestSpec extends AnyFlatSpec {

  "gpt2" should "run SparkNLP pipeline with larger batch size" taggedAs SlowTest in {
    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "My name is Leonardo."),
      (2, "My name is Leonardo and I come from Rome."),
      (3, "My name is"),
      (4, "What is my name?"),
      (5, "Who is Ronaldo?"),
      (6, "Who discovered the microscope?"),
      (7, "Where does petrol come from?"),
      (8, "What is the difference between diesel and petrol?"),
      (9, "Where is Sofia?"),
      (10, "The priest is convinced that"),
    )).toDF("id", "text").repartition(1)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val gpt2 = GPT2Transformer
      .pretrained()
      .setInputCols(Array("documents"))
      .setMaxOutputLength(50)
      .setDoSample(false)
      .setTopK(50)
      .setBatchSize(5)
      .setNoRepeatNgramSize(3)
      .setOutputCol("generation")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, gpt2))

    val model = pipeline.fit(testData)

    Benchmark.time("Time to generate text", true) {
      val results = model.transform(testData)
      results.select("generation.result").show(truncate = false)
    }
  }

  "gpt2" should "run SparkNLP pipeline with doSample=true " taggedAs SlowTest in {
    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "Leonardo Da Vinci invented the wheel?")
    )).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val gpt2 = GPT2Transformer
      .pretrained()
      .setTask("Is it true that")
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setMaxOutputLength(50)
      .setOutputCol("generation")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, gpt2))

    val model = pipeline.fit(testData)

    val dataframe1 = model.transform(testData).select("generation.result").collect().toSeq.head.getAs[Seq[String]](0).head
    println(dataframe1)
    val dataframe2 = model.transform(testData).select("generation.result").collect().toSeq.head.getAs[Seq[String]](0).head
    println(dataframe2)

    assert(!dataframe1.equals(dataframe2))

  }

  "gpt2" should "run SparkNLP pipeline with doSample=true and fixed random seed " taggedAs SlowTest in {
    val testData = ResourceHelper.spark.createDataFrame(Seq(

      (1, "Preheat the oven to.")
    )).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val gpt2 = GPT2Transformer
      .pretrained()
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setMaxOutputLength(50)
      .setRandomSeed(10)
      .setOutputCol("generation")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, gpt2))

    val model = pipeline.fit(testData)

    val dataframe1 = model.transform(testData).select("generation.result").collect().toSeq.head.getAs[Seq[String]](0).head
    println(dataframe1)
    val dataframe2 = model.transform(testData).select("generation.result").collect().toSeq.head.getAs[Seq[String]](0).head
    println(dataframe2)

    assert(dataframe1.equals(dataframe2))
  }


}
