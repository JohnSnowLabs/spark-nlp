package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.BpeTokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File

class GPT2TestSpec  extends AnyFlatSpec {

  "gpt2" should "run SparkNLP pipeline" taggedAs SlowTest in {
    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "My name is Leonardo.")
    )).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val gpt2 = GPT2Transformer
        .loadSavedModel("/models/gpt2/gpt2", spark = ResourceHelper.spark)
        .setInputCols(Array("documents"))
        .setMaxOutputLength(50)
        .setDoSample(false)
        .setTopK(50)
        .setNoRepeatNgramSize(3)
        .setOutputCol("generation")

//    gpt2.write.overwrite.save("/tmp/gpt2")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, gpt2))

    val model = pipeline.fit(testData)
    val results = model.transform(testData)

    results.select("generation.result").show(truncate = false)
  }

  "gpt2" should "run SparkNLP pipeline with doSample=true " taggedAs SlowTest in {
    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "Leonardo Da Vinci invented the wheel?")
    )).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val gpt2 = GPT2Transformer
      .load("/tmp/gpt2")
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
      .load("/tmp/gpt2")
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
