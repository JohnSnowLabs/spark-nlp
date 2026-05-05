/*
 * Copyright 2017-2024 John Snow Labs
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

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.{NGramGenerator, Tokenizer}
import com.johnsnowlabs.nlp.base.{Doc2Chunk, DocumentAssembler}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, EmbeddingsFinisher}
import com.johnsnowlabs.nlp.AnnotationUtils._
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row
import org.scalatest.flatspec.AnyFlatSpec

class LateChunkEmbeddingsTestSpec extends AnyFlatSpec {

  "LateChunkEmbeddings" should "produce SENTENCE_EMBEDDINGS with AVERAGE pooling" taggedAs FastTest in {

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    // GLove provides WORD_EMBEDDINGS over the full document (sentence id = 0)
    val embeddings = AnnotatorBuilder
      .getGLoveEmbeddings(smallCorpus)
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("chunk")
      .setN(2)

    val lateChunkEmbeddings = new LateChunkEmbeddings()
      .setInputCols(Array("document", "chunk", "embeddings"))
      .setOutputCol("late_chunk_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, nGrams, lateChunkEmbeddings))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)

    val annotations = Annotation.collect(pipelineDF, "late_chunk_embeddings").flatten

    // Every output annotation must carry the SENTENCE_EMBEDDINGS type
    assert(
      annotations.forall(_.annotatorType == SENTENCE_EMBEDDINGS),
      "All output annotations should have annotatorType == SENTENCE_EMBEDDINGS")

    // Embeddings must be non-empty and of consistent dimension
    assert(annotations.nonEmpty, "There should be at least one chunk embedding")
    val dims = annotations.map(_.embeddings.length).distinct
    assert(dims.length == 1, s"All chunk embeddings should have the same dimension, got: $dims")
    assert(dims.head > 0, "Embedding dimension must be positive")
  }

  "LateChunkEmbeddings" should "preserve chunk metadata in output annotations" taggedAs FastTest in {

    val document = "Record: Bush Blue, ZIPCODE: XYZ84556222, phone: (911) 45 88".toRow()

    val chunks = Row(
      Seq(
        Annotation(
          CHUNK,
          8,
          16,
          "Bush Blue",
          Map("entity" -> "NAME", "sentence" -> "0", "chunk" -> "0", "confidence" -> "0.98"))
          .toRow(),
        Annotation(
          CHUNK,
          48,
          58,
          "(911) 45 88",
          Map("entity" -> "PHONE", "sentence" -> "0", "chunk" -> "1", "confidence" -> "1.0"))
          .toRow()))

    val df = createAnnotatorDataframe("sentence", DOCUMENT, document)
      .crossJoin(createAnnotatorDataframe("chunk", CHUNK, chunks))

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val wordEmbeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val lateChunkEmbeddings = new LateChunkEmbeddings()
      .setInputCols("sentence", "chunk", "embeddings")
      .setOutputCol("late_chunk_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pipeline = new Pipeline().setStages(Array(tokenizer, wordEmbeddings, lateChunkEmbeddings))
    val resultDf = pipeline.fit(df).transform(df)

    val annotations = Annotation.collect(resultDf, "late_chunk_embeddings").flatten

    // Exactly one output annotation per input chunk
    assert(annotations.length == 2, s"Expected 2 annotations, got ${annotations.length}")

    // Custom metadata fields from the input CHUNK are forwarded to the output
    assert(annotations(0).metadata("entity") == "NAME")
    assert(annotations(1).metadata("entity") == "PHONE")

    // Standard Late-chunking metadata keys must always be present
    val requiredKeys =
      Set("entity", "sentence", "chunk", "confidence", "token", "pieceId", "isWordStart")
    assert(
      annotations.forall(a => requiredKeys.forall(a.metadata.contains)),
      s"Some required metadata keys are missing. Got: ${annotations.map(_.metadata.keys.toSet)}")

    // Output type is SENTENCE_EMBEDDINGS
    assert(annotations.forall(_.annotatorType == SENTENCE_EMBEDDINGS))
  }

  "LateChunkEmbeddings" should "drop chunk annotations that have no overlapping token embeddings" taggedAs FastTest in {

    // Chunk span [999, 1020] is far outside the short document — no tokens will match
    val document = "Short text here.".toRow()

    val chunks = Row(
      Seq(
        Annotation(CHUNK, 0, 4, "Short", Map("sentence" -> "0", "chunk" -> "0"))
          .toRow(),
        Annotation(CHUNK, 999, 1020, "out of range", Map("sentence" -> "0", "chunk" -> "1"))
          .toRow()))

    val df = createAnnotatorDataframe("sentence", DOCUMENT, document)
      .crossJoin(createAnnotatorDataframe("chunk", CHUNK, chunks))

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val wordEmbeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val lateChunkEmbeddings = new LateChunkEmbeddings()
      .setInputCols("sentence", "chunk", "embeddings")
      .setOutputCol("late_chunk_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pipeline = new Pipeline().setStages(Array(tokenizer, wordEmbeddings, lateChunkEmbeddings))
    val resultDf = pipeline.fit(df).transform(df)

    val annotations = Annotation.collect(resultDf, "late_chunk_embeddings").flatten

    // The valid chunk ("Short") should produce an embedding; the OOR chunk must be silently dropped
    assert(
      annotations.length == 1,
      s"Expected 1 annotation for the valid chunk, got ${annotations.length}")
    assert(annotations(0).result == "Short")
  }

  "LateChunkEmbeddings" should "support SUM pooling strategy" taggedAs FastTest in {

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AnnotatorBuilder
      .getGLoveEmbeddings(smallCorpus)
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("chunk")
      .setN(2)

    val lateChunkEmbeddings = new LateChunkEmbeddings()
      .setInputCols(Array("document", "chunk", "embeddings"))
      .setOutputCol("late_chunk_embeddings")
      .setPoolingStrategy("SUM")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, nGrams, lateChunkEmbeddings))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    val annotations = Annotation.collect(pipelineDF, "late_chunk_embeddings").flatten

    assert(annotations.nonEmpty)
    assert(annotations.forall(_.annotatorType == SENTENCE_EMBEDDINGS))
    assert(annotations.forall(_.embeddings.nonEmpty))
  }

  "LateChunkEmbeddings" should "feed into EmbeddingsFinisher without errors" taggedAs FastTest in {

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AnnotatorBuilder
      .getGLoveEmbeddings(smallCorpus)
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("chunk")
      .setN(2)

    val lateChunkEmbeddings = new LateChunkEmbeddings()
      .setInputCols(Array("document", "chunk", "embeddings"))
      .setOutputCol("late_chunk_embeddings")
      .setPoolingStrategy("AVERAGE")

    val finisher = new EmbeddingsFinisher()
      .setInputCols("late_chunk_embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(
        Array(documentAssembler, tokenizer, embeddings, nGrams, lateChunkEmbeddings, finisher))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)

    // EmbeddingsFinisher must produce a non-empty result column
    val finishedCount = pipelineDF
      .selectExpr("explode(finished_embeddings) as e")
      .count()
    assert(finishedCount > 0, "EmbeddingsFinisher produced no output from LateChunkEmbeddings")
  }

  "LateChunkEmbeddings" should
    "produce correct SENTENCE_EMBEDDINGS end-to-end with LongformerEmbeddings" taggedAs SlowTest in {

      import ResourceHelper.spark.implicits._

      // Two-sentence medical document; the second sentence references context from the first.
      // With late chunking the pooled embedding for chunk 1 ("It caused severe nausea...") should
      // carry signal from the full document, not just from that clause in isolation.
      val data = Seq(
        (
          "AcmeDrug was prescribed for migraine in March. The patient took two doses.\n\n" +
            "It caused severe nausea the next day, and therapy was stopped.",
          Array(
            "AcmeDrug was prescribed for migraine in March. The patient took two doses.",
            "It caused severe nausea the next day, and therapy was stopped.")))
        .toDF("text", "chunks")

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      // LongformerEmbeddings processes the FULL document in a single forward pass over up to
      // 4 096 tokens: exactly what late chunking requires.
      // Default pretrained model: "longformer_base_4096" (768-dim, English).
      val longformer = LongformerEmbeddings
        .pretrained()
        .setInputCols("document", "token")
        .setOutputCol("token_embeddings")
        .setCaseSensitive(true)
        .setMaxSentenceLength(512)

      val chunker = new Doc2Chunk()
        .setInputCols(Array("document"))
        .setChunkCol("chunks")
        .setIsArray(true)
        .setOutputCol("chunk")

      val lateChunkEmbeddings = new LateChunkEmbeddings()
        .setInputCols("document", "chunk", "token_embeddings")
        .setOutputCol("late_chunk_embeddings")
        .setPoolingStrategy("AVERAGE")

      val pipeline = new Pipeline()
        .setStages(Array(documentAssembler, tokenizer, longformer, chunker, lateChunkEmbeddings))

      val result = pipeline.fit(data).transform(data)

      val annotations = Annotation.collect(result, "late_chunk_embeddings").flatten

      // Exactly two chunks → exactly two output annotations
      assert(
        annotations.length == 2,
        s"Expected 2 annotations (one per chunk), got ${annotations.length}")

      // Both must carry SENTENCE_EMBEDDINGS type
      assert(
        annotations.forall(_.annotatorType == SENTENCE_EMBEDDINGS),
        "All output annotations should have annotatorType == SENTENCE_EMBEDDINGS")

      // Longformer base dim = 768; embeddings must be non-trivially sized
      val dim = annotations.head.embeddings.length
      assert(dim > 0, "Embedding dimension should be positive")

      // All embeddings must share the same dimension
      assert(
        annotations.forall(_.embeddings.length == dim),
        s"All chunk embeddings should have dimension $dim")

      // The schema metadata dimension written by afterAnnotate must match the actual vector length
      val schemaDim =
        result.schema(lateChunkEmbeddings.getOutputCol).metadata.getLong("dimension")
      assert(
        schemaDim == dim,
        s"Schema metadata dimension ($schemaDim) does not match actual embedding size ($dim)")

      // Chunk result text must be preserved verbatim
      assert(annotations(0).result.startsWith("AcmeDrug"))
      assert(annotations(1).result.startsWith("It caused"))

      result
        .selectExpr("explode(late_chunk_embeddings) as r")
        .select("r.annotatorType", "r.result", "r.embeddings")
        .show(5, 80)
    }

  "LateChunkEmbeddings" should
    "be usable downstream by EmbeddingsFinisher when fed by LongformerEmbeddings" taggedAs SlowTest in {

      import ResourceHelper.spark.implicits._

      val data = Seq(
        (
          "The patient was treated with AcmeDrug. Side effects included nausea and fatigue.",
          Array(
            "The patient was treated with AcmeDrug.",
            "Side effects included nausea and fatigue."))).toDF("text", "chunks")

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val longformer = LongformerEmbeddings
        .pretrained()
        .setInputCols("document", "token")
        .setOutputCol("token_embeddings")
        .setCaseSensitive(true)
        .setMaxSentenceLength(512)

      val chunker = new Doc2Chunk()
        .setInputCols(Array("document"))
        .setChunkCol("chunks")
        .setIsArray(true)
        .setOutputCol("chunk")

      val lateChunkEmbeddings = new LateChunkEmbeddings()
        .setInputCols("document", "chunk", "token_embeddings")
        .setOutputCol("late_chunk_embeddings")
        .setPoolingStrategy("AVERAGE")

      val finisher = new EmbeddingsFinisher()
        .setInputCols("late_chunk_embeddings")
        .setOutputCols("finished_embeddings")
        .setOutputAsVector(true)
        .setCleanAnnotations(false)

      val pipeline = new Pipeline()
        .setStages(
          Array(documentAssembler, tokenizer, longformer, chunker, lateChunkEmbeddings, finisher))

      val result = pipeline.fit(data).transform(data)

      val finishedCount = result.selectExpr("explode(finished_embeddings) as e").count()
      assert(
        finishedCount == 2,
        s"Expected 2 finished embeddings vectors (one per chunk), got $finishedCount")
    }

}
