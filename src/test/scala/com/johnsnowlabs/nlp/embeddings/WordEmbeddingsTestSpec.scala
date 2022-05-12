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

import com.johnsnowlabs.nlp.AnnotatorType.WORD_EMBEDDINGS
import com.johnsnowlabs.nlp.annotators.{SparkSessionTest, Tokenizer}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

class WordEmbeddingsTestSpec extends AnyFlatSpec with SparkSessionTest {

  "Word Embeddings" should "correctly embed clinical words not embed non-existent words" taggedAs SlowTest in {

    val words = spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/clinical_words.txt")
    val notWords = spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/not_words.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("word")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings))

    val wordsP = pipeline.fit(words).transform(words).cache()
    val notWordsP = pipeline.fit(notWords).transform(notWords).cache()

    val wordsCoverage =
      WordEmbeddingsModel.withCoverageColumn(wordsP, "embeddings", "cov_embeddings")
    val notWordsCoverage =
      WordEmbeddingsModel.withCoverageColumn(notWordsP, "embeddings", "cov_embeddings")

    wordsCoverage.select("word", "cov_embeddings").show(1)
    notWordsCoverage.select("word", "cov_embeddings").show(1)

    val wordsOverallCoverage =
      WordEmbeddingsModel.overallCoverage(wordsCoverage, "embeddings").percentage
    val notWordsOverallCoverage =
      WordEmbeddingsModel.overallCoverage(notWordsCoverage, "embeddings").percentage

    spark
      .createDataFrame(
        Seq(("Words", wordsOverallCoverage), ("Not Words", notWordsOverallCoverage)))
      .toDF("Dataset", "OverallCoverage")
      .show(1)

    assert(wordsOverallCoverage == 1)
    assert(notWordsOverallCoverage == 0)
  }

  "Word Embeddings" should "store and load from disk" taggedAs FastTest in {

    val data = spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/clinical_words.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("word")
      .setOutputCol("document")

    val embeddings = new WordEmbeddings()
      .setStoragePath("src/test/resources/random_embeddings_dim4.txt", ReadAs.TEXT)
      .setDimension(4)
      .setStorageRef("glove_4d")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings))

    val model = pipeline.fit(data)

    model.write.overwrite().save("./tmp_embeddings_pipeline")

    val loadedPipeline = PipelineModel.load("./tmp_embeddings_pipeline")
    val embeddingsDataset = loadedPipeline.transform(data)

    val expectedEmbeddings = getExpectedEmbeddings
    val actualEmbeddings = AssertAnnotations.getActualResult(embeddingsDataset, "embeddings")
    AssertAnnotations.assertFields(expectedEmbeddings, actualEmbeddings)
  }

  "Word Embeddings with in-memory" should "store and load from disk" taggedAs FastTest in {

    val data = spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/clinical_words.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("word")
      .setOutputCol("document")

    val embeddings = new WordEmbeddings()
      .setStoragePath("src/test/resources/random_embeddings_dim4.txt", ReadAs.TEXT)
      .setDimension(4)
      .setStorageRef("glove_4d")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setEnableInMemoryStorage(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

    val model = pipeline.fit(data)

    model.write.overwrite().save("./tmp_embeddings_pipeline")
    val loadedPipeline = PipelineModel.load("./tmp_embeddings_pipeline")
    val embeddingsDataset = loadedPipeline.transform(data)

    val expectedEmbeddings = getExpectedEmbeddings
    val actualEmbeddings = AssertAnnotations.getActualResult(embeddingsDataset, "embeddings")
    AssertAnnotations.assertFields(expectedEmbeddings, actualEmbeddings)
  }

  private def getExpectedEmbeddings: Array[Seq[Annotation]] = {
    val expectedEmbeddings = Array(
      Seq(
        Annotation(
          WORD_EMBEDDINGS,
          0,
          8,
          "diagnosis",
          Map(
            "isOOV" -> "false",
            "pieceId" -> "-1",
            "isWordStart" -> "true",
            "token" -> "diagnosis",
            "sentence" -> "0"),
          Array(0.9076976f, 0.13794145f, 0.7322122f, 0.37095428f))),
      Seq(
        Annotation(
          WORD_EMBEDDINGS,
          0,
          6,
          "obesity",
          Map(
            "isOOV" -> "true",
            "pieceId" -> "-1",
            "isWordStart" -> "true",
            "token" -> "obesity",
            "sentence" -> "0"),
          Array(0.0f, 0.0f, 0.0f, 0.0f))),
      Seq(
        Annotation(
          WORD_EMBEDDINGS,
          0,
          7,
          "diabetes",
          Map(
            "isOOV" -> "false",
            "pieceId" -> "-1",
            "isWordStart" -> "true",
            "token" -> "diabetes",
            "sentence" -> "0"),
          Array(0.5955276f, 0.01899012f, 0.43977284f, 0.8911282f))),
      Seq(
        Annotation(
          WORD_EMBEDDINGS,
          0,
          8,
          "chlamydia",
          Map(
            "isOOV" -> "true",
            "pieceId" -> "-1",
            "isWordStart" -> "true",
            "token" -> "chlamydia",
            "sentence" -> "0"),
          Array(0.0f, 0.0f, 0.0f, 0.0f))))

    expectedEmbeddings
  }

}
