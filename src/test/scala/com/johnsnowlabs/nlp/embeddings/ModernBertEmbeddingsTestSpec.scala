/*
 * Copyright 2017-2025 John Snow Labs
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

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.{EmbeddingsFinisher, AnnotatorType}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class ModernBertEmbeddingsTestSpec extends AnyFlatSpec {

  "ModernBert Embeddings" should "correctly embed tokens in a sentence" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Something is weird on this text",
      "ModernBERT is a modern bidirectional encoder model.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = ModernBertEmbeddings
      .pretrained("modernbert-base", "en")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(512)

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(document, tokenizer, embeddings, embeddingsFinisher))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("finished_embeddings").show()

  }

  "ModernBert Embeddings" should "correctly work with empty tokens" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = ModernBertEmbeddings
      .pretrained("modernbert-base", "en")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(512)

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(document, tokenizer, embeddings, embeddingsFinisher))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("finished_embeddings").show()

  }

  "ModernBert Embeddings" should "handle special characters" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("This is a test with @#$%^&*() special characters!").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = ModernBertEmbeddings
      .pretrained("modernbert-base", "en")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(document, tokenizer, embeddings, embeddingsFinisher))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("finished_embeddings").show()

  }

  "ModernBert Embeddings" should "work with long sentences" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val longText =
      "This is a very long sentence that should test the capabilities of ModernBERT with extended context. " * 50
    val ddd = Seq(longText).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = ModernBertEmbeddings
      .pretrained("modernbert-base", "en")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setMaxSentenceLength(4096) // Test with longer context

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(document, tokenizer, embeddings, embeddingsFinisher))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("finished_embeddings").show()

  }

  "ModernBert Embeddings" should "benchmark batches" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Something is weird on this text",
      "ModernBERT is a modern bidirectional encoder model.",
      "The quick brown fox jumps over the lazy dog.",
      "I love using Spark NLP for natural language processing tasks.",
      "Machine learning is revolutionizing how we process text data.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = ModernBertEmbeddings
      .pretrained("modernbert-base", "en")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setBatchSize(2)

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(document, tokenizer, embeddings, embeddingsFinisher))

    val pipelineModel = pipeline.fit(ddd)

    Benchmark.time("Time to save ModernBertEmbeddings results") {
      pipelineModel.transform(ddd).write.mode("overwrite").parquet("./tmp_modernbert")
    }

  }

  "ModernBertEmbeddings" should "correctly set parameters" in {
    val embeddings = ModernBertEmbeddings
      .pretrained("modernbert-base", "en")
      .setMaxSentenceLength(512)
      .setCaseSensitive(true)
      .setBatchSize(4)
      .setDimension(768)

    assert(embeddings.getMaxSentenceLength == 512)
    assert(embeddings.getCaseSensitive)
    assert(embeddings.getBatchSize == 4)
    assert(embeddings.getDimension == 768)
  }

  "ModernBertEmbeddings" should "fail when maxSentenceLength is too large" in {
    assertThrows[IllegalArgumentException] {
      ModernBertEmbeddings
        .pretrained("modernbert-base", "en")
        .setMaxSentenceLength(8193) // Too large, should fail
    }
  }

  "ModernBertEmbeddings" should "fail when maxSentenceLength is too small" in {
    assertThrows[IllegalArgumentException] {
      ModernBertEmbeddings
        .pretrained("modernbert-base", "en")
        .setMaxSentenceLength(0) // Too small, should fail
    }
  }

  "ModernBertEmbeddings" should "have correct input and output types" in {
    val embeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en")

    assert(
      embeddings.inputAnnotatorTypes.sameElements(
        Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)))
    assert(embeddings.outputAnnotatorType == AnnotatorType.WORD_EMBEDDINGS)
  }
}
