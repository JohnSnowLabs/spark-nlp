/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, size}
import org.scalatest.flatspec.AnyFlatSpec

class BGEEmbeddingsTestSpec extends AnyFlatSpec {

  "BGE Embeddings" should "correctly embed multiple sentences" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "query: how much protein should a female eat",
      "query: summit define",
      "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 " +
        "grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or" +
        " training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
      "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of" +
        " a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more" +
        " governments.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = BGEEmbeddings
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("bge")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(ddd).transform(ddd)
    pipelineDF.select("bge.embeddings").show(truncate = false)

  }

  "BGE Embeddings" should "be saved and loaded correctly" taggedAs SlowTest in {


    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "query: how much protein should a female eat",
      "query: summit define",
      "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 " +
        "grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or" +
        " training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
      "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of" +
        " a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more" +
        " governments.")
      .toDF("text")


    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = BGEEmbeddings
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, embeddings))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("embeddings.result").show(false)

    Benchmark.time("Time to save BGEEmbeddings pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_bge_pipeline")
    }

    Benchmark.time("Time to save BGEEmbeddings model") {
      pipelineModel.stages.last
        .asInstanceOf[BGEEmbeddings]
        .write
        .overwrite()
        .save("./tmp_bge_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_bge_pipeline")
    loadedPipelineModel.transform(ddd).select("embeddings.result").show(false)

    val loadedSequenceModel = BGEEmbeddings.load("./tmp_bge_model")

  }

  it should "have embeddings of the same size" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._
    val testDf = Seq(
      "I like apples",
      "I like bananas \\n and other things \\n like icream \\n and cats",
      "I like rockets")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = BGEEmbeddings
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("bge")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(testDf).transform(testDf)

    val embeddingsDF = pipelineDF.withColumn("embeddings", col("bge.embeddings").getItem(0))

    val sizesArray: Array[Int] = embeddingsDF
      .select(size(col("embeddings")).as("size"))
      .collect()
      .map(row => row.getAs[Int]("size"))

    assert(sizesArray.forall(_ == sizesArray.head))
  }

  it should "work with sentences" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._
    val testData = "I really enjoy my job. This is amazing"
    val testDf = Seq(testData).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetectorDL = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "en")
      .setInputCols(Array("document"))
      .setOutputCol("sentences")

    val embeddings = BGEEmbeddings
      .pretrained()
      .setInputCols(Array("sentences"))
      .setOutputCol("bge")

    val pipeline = new Pipeline().setStages(Array(document, sentenceDetectorDL, embeddings))

    val pipelineDF = pipeline.fit(testDf).transform(testDf)
    pipelineDF.select("bge.embeddings").show(false)
  }

  it should "not return empty embeddings" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._
    val interests = Seq(
      "I like music",
      "I like movies",
      "I like books",
      "I like sports",
      "I like travel",
      "I like food",
      "I like games",
      "I like art",
      "I like nature",
      "I like science",
      "I like technology",
      "I like history",
      "I like fashion",
      "I like cars",
      "I like animals",
      "I like gardening")
    val testDf = interests.toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = BGEEmbeddings
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("bge")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(testDf).transform(testDf)

    val embeddingsDF = pipelineDF.withColumn("embeddings", col("bge.embeddings").getItem(0))

    val sizesArray: Array[Int] = embeddingsDF
      .select(size(col("embeddings")).as("size"))
      .collect()
      .map(row => row.getAs[Int]("size"))

    assert(sizesArray.forall(_ > 0))
  }

}
