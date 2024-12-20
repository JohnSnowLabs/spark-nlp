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

import com.johnsnowlabs.nlp.annotators.{StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

class CamemBertEmbeddingsTestSpec extends AnyFlatSpec {

  "CamemBertEmbeddings" should "correctly work with empty tokens" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")
      .limit(50)


    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(
        Array("this", "is", "my", "document", "sentence", "second", "first", ",", "."))
      .setCaseSensitive(false)

    val embeddings = CamemBertEmbeddings
      .pretrained()
      .setInputCols("document", "cleanTokens")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, stopWordsCleaner, embeddings))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    Benchmark.time("Time to save CamemBertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_embeddings")
    }
  }

  "CamemBertEmbeddings" should "be saved and loaded correctly" taggedAs SlowTest in {


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

    val embeddings = CamemBertEmbeddings
      .pretrained()
      .setInputCols("document","token")
      .setOutputCol("embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("embeddings.result").show(false)

    Benchmark.time("Time to save CamemBertEmbeddings pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_camembert_pipeline")
    }

    Benchmark.time("Time to save CamemBertEmbeddings model") {
      pipelineModel.stages.last
        .asInstanceOf[CamemBertEmbeddings]
        .write
        .overwrite()
        .save("./tmp_camembert_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_camembert_pipeline")
    loadedPipelineModel.transform(ddd).select("embeddings.result").show(false)

    val loadedSequenceModel = CamemBertEmbeddings.load("./tmp_camembert_model")

  }



  "CamemBertEmbeddings" should "benchmark test" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._
    import ResourceHelper.spark.implicits._

    val conll = CoNLL(explodeSentences = false)
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")
        .limit(50)

    val embeddings = CamemBertEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)
      .setBatchSize(16)

    val pipeline = new Pipeline()
      .setStages(Array(embeddings))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save CamemBertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_embeddings")
    }

    println("missing tokens/embeddings: ")
    pipelineDF
      .withColumn("sentence_size", size(col("sentence")))
      .withColumn("token_size", size(col("token")))
      .withColumn("embed_size", size(col("embeddings")))
      .where(col("token_size") =!= col("embed_size"))
      .select("sentence_size", "token_size", "embed_size", "token.result", "embeddings.result")
      .show(false)

    println("total sentences: ", pipelineDF.select(explode($"sentence.result")).count)
    val totalTokens = pipelineDF.select(explode($"token.result")).count.toInt
    val totalEmbeddings = pipelineDF.select(explode($"embeddings.embeddings")).count.toInt

    println(s"total tokens: $totalTokens")
    println(s"total embeddings: $totalEmbeddings")

    assert(totalTokens == totalEmbeddings)
  }

  "CamemBertEmbeddings" should "download, save, and load a model" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("This is just a simple sentence for the testing purposes!").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = CamemBertEmbeddings
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)
      .setBatchSize(12)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    val pipelineModel = pipeline.fit(ddd)
    pipelineModel.transform(ddd).show()

    Benchmark.time("Time to save CamemBertEmbeddings pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_camembert_pipeline")
    }

    Benchmark.time("Time to save RoBertaEmbeddings model") {
      pipelineModel.stages.last
        .asInstanceOf[CamemBertEmbeddings]
        .write
        .overwrite()
        .save("./tmp_camembert_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_camembert_pipeline")
    loadedPipelineModel.transform(ddd).show()

    val loadedCamemBertModel = CamemBertEmbeddings.load("./tmp_camembert_model")
    loadedCamemBertModel.getDimension

  }

  "CamemBertEmbeddings" should "be aligned with custom tokens from Tokenizer" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Paris est la capitale de la France .",
      "Le camembert est délicieux .",
      "Le camembert est excellent !",
      " J'aime le camembert ! .  . .",
      "\\u2009.Les modèles de langue contextuels Camembert pour le Français : impact de la taille et de l'hétérogénéité des données d'entrainement .\\u2009.\\u2009.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = CamemBertEmbeddings
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)
      .setBatchSize(12)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    println("tokens: ")
    pipelineDF.select("token.result").show(false)
    println("embeddings: ")
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
