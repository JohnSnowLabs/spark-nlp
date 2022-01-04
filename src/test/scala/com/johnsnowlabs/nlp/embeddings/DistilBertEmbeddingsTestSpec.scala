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

import com.johnsnowlabs.nlp.annotators.{SparkSessionTest, StopWordsCleaner}
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

class DistilBertEmbeddingsTestSpec extends AnyFlatSpec with SparkSessionTest {


  "DistilBertEmbeddings" should "correctly work with empty tokens" taggedAs SlowTest in {

    val smallCorpus = spark.read.option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(Array("this", "is", "my", "document", "sentence", "second", "first", ",", "."))
      .setCaseSensitive(false)

    val embeddings = DistilBertEmbeddings.pretrained()
      .setInputCols("document", "cleanTokens")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        stopWordsCleaner,
        embeddings
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
    }
  }

  "DistilBertEmbeddings" should "benchmark test" taggedAs SlowTest in {
    import spark.implicits._

    val conll = CoNLL()
    val training_data = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")
    println(training_data.count())

    val embeddings = DistilBertEmbeddings.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)
      .setBatchSize(16)

    val pipeline = new Pipeline()
      .setStages(Array(
        embeddings
      ))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save DistilBertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
    }

    println("missing tokens/embeddings: ")
    pipelineDF.withColumn("sentence_size", size(col("sentence")))
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

  "DistilBertEmbeddings" should "download, save, and load a model" taggedAs SlowTest in {

    import spark.implicits._

    val ddd = Seq(
      "This is just a simple sentence for the testing purposes!"
    ).toDF("text")

    val embeddings = DistilBertEmbeddings.pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)
      .setBatchSize(16)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

    val pipelineModel = pipeline.fit(ddd)
    pipelineModel.transform(ddd).show()

    Benchmark.time("Time to save DistilBertEmbeddings pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_distilbert_pipeline")
    }

    Benchmark.time("Time to save DistilBertEmbeddings model") {
      pipelineModel
        .stages.last
        .asInstanceOf[DistilBertEmbeddings]
        .write.overwrite()
        .save("./tmp_distilbert_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_distilbert_pipeline")
    loadedPipelineModel.transform(ddd).show()

    val loadedDistilBertModel = DistilBertEmbeddings.load("./tmp_distilbert_model")
    loadedDistilBertModel.getDimension

  }

  "DistilBert Embeddings" should "infer with Pytorch load model" taggedAs SlowTest ignore {
    //TODO: Load pretrained python model enable this test
    import spark.implicits._

    val dataFrame = Seq("Peter lives in New York", "Jon Snow lives in Winterfell").toDS().toDF("text")
    val distilBert = DistilBertEmbeddings.load("./tmp_distilbert_base_pt")
      .setInputCols("document", "token")
      .setOutputCol("distilbert")
      .setCaseSensitive(true)
      .setDeepLearningEngine("pytorch")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, distilBert))
    val pipelineDF = pipeline.fit(dataFrame).transform(dataFrame)

    val embeddingsDF = pipelineDF.select($"distilbert.embeddings"(0))

    assert(!embeddingsDF.isEmpty)
  }

  "DistilBert Embeddings" should "raise an error when setting a not supported deep learning engine" taggedAs SlowTest in {
    import spark.implicits._
    val dataFrame = Seq("Peter lives in New York", "Jon Snow lives in Winterfell").toDS().toDF("text")
    val distilBert = DistilBertEmbeddings.pretrained()
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

  "DistilBert Embeddings" should "raise an error when setting a deep learning engine different from pre-trained model" taggedAs SlowTest in {
    import spark.implicits._

    val dataFrame = Seq("Peter lives in New York", "Jon Snow lives in Winterfell").toDS().toDF("text")
    val distilBert = DistilBertEmbeddings.pretrained()
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

  "DistilBert Embeddings" should "raise an error when sentence length is not between 1 and 512" taggedAs SlowTest in {

    var expectedErrorMessage = "requirement failed: " +
      "DISTILBERT models do not support sequences longer than 512 because of trainable positional embeddings"

    var caught = intercept[IllegalArgumentException] {
      DistilBertEmbeddings.pretrained()
        .setInputCols("document", "token")
        .setOutputCol("distilbert")
        .setMaxSentenceLength(513)
    }

    assert(caught.getMessage == expectedErrorMessage)

    expectedErrorMessage = "requirement failed: " + "The maxSentenceLength must be at least 1"

    caught = intercept[IllegalArgumentException] {
      DistilBertEmbeddings.pretrained()
        .setInputCols("document", "token")
        .setOutputCol("distilbert")
        .setMaxSentenceLength(-2)
    }

    assert(caught.getMessage == expectedErrorMessage)

  }

}
