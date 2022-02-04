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

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec


class AlbertEmbeddingsTestSpec extends AnyFlatSpec with SparkSessionTest {

  "AlbertEmbeddings" should "correctly load pretrained model" taggedAs SlowTest in {

    val smallCorpus = spark.read.option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        embeddings
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    pipelineDF.select("token.result").show(1, truncate = false)
    pipelineDF.select("embeddings.result").show(1, truncate = false)
    pipelineDF.select("embeddings.metadata").show(1, truncate = false)
    pipelineDF.select("embeddings.embeddings").show(1, truncate = 300)
    pipelineDF.select(size(pipelineDF("embeddings.embeddings")).as("embeddings_size")).show(1)
    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.select("embeddings").write.mode("overwrite").parquet("./tmp_albert_embeddings")
    }
  }

  "AlbertEmbeddings" should "benchmark test" taggedAs SlowTest in {
    import spark.implicits._

    val conll = CoNLL()
    val training_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val embeddings = AlbertEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline()
      .setStages(Array(
        embeddings
      ))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save AlbertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
    }

    Benchmark.time("Time to finish checking counts in results") {
      println("missing tokens/embeddings: ")
      pipelineDF.withColumn("sentence_size", size(col("sentence")))
        .withColumn("token_size", size(col("token")))
        .withColumn("embed_size", size(col("embeddings")))
        .where(col("token_size") =!= col("embed_size"))
        .select("sentence_size", "token_size", "embed_size")
        .show(false)
    }

    Benchmark.time("Time to finish explod/count in results") {
      println("total sentences: ", pipelineDF.select(explode($"sentence.result")).count)
      val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
      val totalEmbeddings = pipelineDF.select(explode($"embeddings.embeddings")).count.toInt

      println(s"total tokens: $totalTokens")
      println(s"total embeddings: $totalEmbeddings")

      // it is normal that the embeddings is less than total tokens in a sentence/document
      // tokens generate multiple sub-wrods or pieces which won't be included in the final results
      assert(totalTokens >= totalEmbeddings)

    }
  }

  "AlbertEmbeddings" should "be aligned with custom tokens from Tokenizer" taggedAs SlowTest in {

    import spark.implicits._

    val ddd = Seq(
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21",
      " carbon emissions have come down without impinging on our growth . . .",
      "carbon emissions have come down without impinging on our growth .\\u2009.\\u2009."
    ).toDF("text")

    val embeddings = AlbertEmbeddings
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("token").show(false)
    pipelineDF.select("embeddings.result").show(false)
    pipelineDF
      .withColumn("token_size", size(col("token")))
      .withColumn("embed_size", size(col("embeddings")))
      .where(col("token_size") =!= col("embed_size"))
      .select("token_size", "embed_size", "token.result", "embeddings.result")
      .show(false)

    val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
    val totalEmbeddings = pipelineDF.select(explode($"embeddings.embeddings")).count.toInt

    println(s"total tokens: $totalTokens")
    println(s"total embeddings: $totalEmbeddings")

    assert(totalTokens == totalEmbeddings)

  }

  "AlbertEmbeddings" should "predict with PyTorch model" taggedAs SlowTest in {
    //TODO: Load pretrained python model enable this test
    import spark.implicits._

    val dataFrame = Seq("Peter lives in New York", "Jon Snow lives in Winterfell").toDS().toDF("text")
    val bert = AlbertEmbeddings.load("./tmp_albert_base_pt")
      .setInputCols("document", "token")
      .setOutputCol("albert")
      .setCaseSensitive(true)
      .setDeepLearningEngine("pytorch")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, bert))
    val pipelineDF = pipeline.fit(dataFrame).transform(dataFrame)

    val embeddingsDF = pipelineDF.select($"albert.embeddings"(0))

    assert(!embeddingsDF.isEmpty)
  }

  "AlbertEmbeddings" should "raise an error when setting a not supported deep learning engine" taggedAs SlowTest in {
    import spark.implicits._
    val dataFrame = Seq("Peter lives in New York", "Jon Snow lives in Winterfell").toDS().toDF("text")
    val distilBert = AlbertEmbeddings.pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setDeepLearningEngine("mxnet")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, distilBert))
    val pipelineDF = pipeline.fit(dataFrame).transform(dataFrame)
    val expectedErrorMessage = "Deep learning engine mxnet not supported"

    val caught = intercept[Exception] {
      pipelineDF.collect()
    }

    assert(caught.getMessage.contains(expectedErrorMessage))
  }

  "AlbertEmbeddings" should "raise an error when setting a deep learning engine different from pre-trained model" taggedAs SlowTest in {
    import spark.implicits._

    val dataFrame = Seq("Peter lives in New York", "Jon Snow lives in Winterfell").toDS().toDF("text")
    val distilBert = AlbertEmbeddings.pretrained()
      .setInputCols("document", "token")
      .setOutputCol("distilbert")
      .setCaseSensitive(true)
      .setDeepLearningEngine("pytorch")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, distilBert))
    val pipelineDF = pipeline.fit(dataFrame).transform(dataFrame)
    val expectedErrorMessage = "Pytorch model is empty. Please verify that deep learning engine parameter matches your model."

    val caught = intercept[Exception] {
      pipelineDF.collect()
    }

    assert(caught.getMessage.contains(expectedErrorMessage))
  }

  "AlbertEmbeddings" should "raise an error when sentence length is not between 1 and 512" taggedAs SlowTest in {

    var expectedErrorMessage = "requirement failed: " +
      "ALBERT models do not support sequences longer than 512 because of trainable positional embeddings"

    var caught = intercept[IllegalArgumentException] {
      AlbertEmbeddings.pretrained()
        .setInputCols("document", "token")
        .setOutputCol("albert")
        .setMaxSentenceLength(700)
    }

    assert(caught.getMessage == expectedErrorMessage)

    expectedErrorMessage = "requirement failed: " + "The maxSentenceLength must be at least 1"

    caught = intercept[IllegalArgumentException] {
      AlbertEmbeddings.pretrained()
        .setInputCols("document", "token")
        .setOutputCol("albert")
        .setMaxSentenceLength(-25)
    }

    assert(caught.getMessage == expectedErrorMessage)

  }

}
