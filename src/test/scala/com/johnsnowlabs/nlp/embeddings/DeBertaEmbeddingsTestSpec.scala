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
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec


class DeBertaEmbeddingsTestSpec extends AnyFlatSpec {

  "DeBertaEmbeddings" should "correctly load pretrained model" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val embeddings = DeBertaEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, tokenizer, embeddings))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    pipelineDF.select("token.result").show(1, truncate = false)
    pipelineDF.select("embeddings.result").show(1, truncate = false)
    pipelineDF.select("embeddings.metadata").show(1, truncate = false)
    pipelineDF.select("embeddings.embeddings").show(1, truncate = 300)
    pipelineDF.select(size(pipelineDF("embeddings.embeddings")).as("embeddings_size")).show(1)
    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.select("embeddings").write.mode("overwrite").parquet("./tmp_embeddings")
    }
  }

  "DeBertaEmbeddings" should "be saved and loaded correctly" taggedAs SlowTest in {


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


    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = DeBertaEmbeddings
      .pretrained()
      .setInputCols("document","token")
      .setOutputCol("embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("embeddings.result").show(false)

    Benchmark.time("Time to save DeBertaEmbeddings pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_deberta_pipeline")
    }

    Benchmark.time("Time to save DeBertaEmbeddings model") {
      pipelineModel.stages.last
        .asInstanceOf[DeBertaEmbeddings]
        .write
        .overwrite()
        .save("./tmp_deberta_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_deberta_pipeline")
    loadedPipelineModel.transform(ddd).select("embeddings.result").show(false)

    val loadedSequenceModel = DeBertaEmbeddings.load("./tmp_deberta_model")

  }
  "DeBertaEmbeddings" should "benchmark test" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val conll = CoNLL(explodeSentences = false)
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")
        .limit(50)

    val embeddings = DeBertaEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline()
      .setStages(Array(embeddings))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save DeBertaEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_debert_embeddings")
    }

    Benchmark.time("Time to finish checking counts in results") {
      println("missing tokens/embeddings: ")
      pipelineDF
        .withColumn("sentence_size", size(col("sentence")))
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

      assert(totalTokens == totalEmbeddings)

    }
  }

  "DeBertaEmbeddings" should "be aligned with custom tokens from Tokenizer" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21",
      " carbon emissions have come down without impinging on our growth . . .",
      "carbon emissions have come down without impinging on our growth .\\u2009.\\u2009.").toDF(
      "text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = DeBertaEmbeddings
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(128)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

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
}
