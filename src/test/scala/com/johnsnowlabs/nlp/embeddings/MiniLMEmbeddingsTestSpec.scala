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

import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, size}
import org.scalatest.flatspec.AnyFlatSpec

class MiniLMEmbeddingsTestSpec extends AnyFlatSpec {

  "MiniLM Embeddings" should "correctly embed multiple sentences" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "This is a sample sentence for embedding generation.",
      "Another example sentence to demonstrate MiniLM embeddings.",
      "MiniLM is a lightweight and efficient sentence embedding model that can generate text embeddings for various NLP tasks.",
      "The model achieves comparable results with BERT-base while being much smaller and faster.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = MiniLMEmbeddings
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("minilm")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(ddd).transform(ddd)
    pipelineDF.select("minilm.embeddings").show(truncate = false)

  }

  it should "have embeddings of the same size" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._
    val testDf = Seq(
      "I like apples",
      "I like bananas \\n and other things \\n like ice cream \\n and cats",
      "I like rockets")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = MiniLMEmbeddings
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("minilm")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(testDf).transform(testDf)

    val embeddingsDF = pipelineDF.withColumn("embeddings", col("minilm.embeddings").getItem(0))

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

    val embeddings = MiniLMEmbeddings
      .pretrained()
      .setInputCols(Array("sentences"))
      .setOutputCol("minilm")

    val pipeline = new Pipeline().setStages(Array(document, sentenceDetectorDL, embeddings))

    val pipelineDF = pipeline.fit(testDf).transform(testDf)
    pipelineDF.select("minilm.embeddings").show(false)
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

    val embeddings = MiniLMEmbeddings
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("minilm")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(testDf).transform(testDf)

    val embeddingsDF = pipelineDF.withColumn("embeddings", col("minilm.embeddings").getItem(0))

    val sizesArray: Array[Int] = embeddingsDF
      .select(size(col("embeddings")).as("size"))
      .collect()
      .map(row => row.getAs[Int]("size"))

    assert(sizesArray.forall(_ > 0))
  }

}
