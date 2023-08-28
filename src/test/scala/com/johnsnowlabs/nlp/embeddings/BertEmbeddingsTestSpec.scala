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

class BertEmbeddingsTestSpec extends AnyFlatSpec {

  "Bert Embeddings" should "correctly embed tokens and sentences" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("Something is weird on the notebooks, something is happening.").toDF("text")

    val data1 = Seq(
      "In the Seven Kingdoms of Westeros, a soldier of the ancient Night's Watch order survives an attack by supernatural creatures known as the White Walkers, thought until now to be mythical.")
      .toDF("text")

    val data2 = Seq(
      "In King's Landing, the capital, Jon Arryn, the King's Hand, dies under mysterious circumstances.")
      .toDF("text")

    val data3 = Seq(
      "Tyrion makes saddle modifications for Bran that will allow the paraplegic boy to ride.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = BertEmbeddings
      .pretrained("small_bert_L2_128", "en")
      .setInputCols(Array("token", "document"))
      .setOutputCol("bert")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    pipeline.fit(ddd).transform(ddd)
    pipeline.fit(data1).transform(data1)
    pipeline.fit(data2).transform(data2)
    pipeline.fit(data3).transform(data3)

  }

  "Bert Embeddings" should "correctly work with empty tokens" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

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

    val embeddings = BertEmbeddings
      .pretrained("small_bert_L2_128", "en")
      .setInputCols("document", "cleanTokens")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, stopWordsCleaner, embeddings))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
    }
  }

  "Bert Embeddings" should "benchmark test" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val conll = CoNLL(explodeSentences = false)
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val embeddings = BertEmbeddings
      .pretrained("small_bert_L2_128", "en")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)
      .setBatchSize(16)

    val pipeline = new Pipeline()
      .setStages(Array(embeddings))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
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

  "Bert Embeddings" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("Something is weird on the notebooks, something is happening.").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val tfModelPath = "src/test/resources/tf-hub-bert/model"

    val embeddings = BertEmbeddings
      .loadSavedModel(tfModelPath, ResourceHelper.spark)
      .setInputCols(Array("token", "document"))
      .setOutputCol("bert")
      .setStorageRef("tf_hub_bert_test")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    pipeline.fit(ddd).write.overwrite().save("./tmp_bert_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_bert_pipeline")

    pipelineModel.transform(ddd)
  }

  "Bert Embeddings" should "correctly load custom model with OpenVINO" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("Something is weird on the notebooks, something is happening.").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val tfModelPath = "src/test/resources/tf-hub-bert/model"

    val embeddings = BertEmbeddings
      .loadSavedModel(tfModelPath, ResourceHelper.spark, useOpenvino = true)
      .setInputCols(Array("token", "document"))
      .setOutputCol("bert")
      .setStorageRef("ov_bert_test")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    pipeline.fit(ddd).write.overwrite().save("./tmp_bert_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_bert_pipeline")

    pipelineModel.transform(ddd)
  }

  "Bert Embeddings" should "be aligned with custom tokens from Tokenizer" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21",
      " carbon emissions have come down without impinging on our growth. .  . .",
      "\\u2009.carbon emissions have come down without impinging on our growth .\\u2009.\\u2009.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = BertEmbeddings
      .pretrained("small_bert_L2_128")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)
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
