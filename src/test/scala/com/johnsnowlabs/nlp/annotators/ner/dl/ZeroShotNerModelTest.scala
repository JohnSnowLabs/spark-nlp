/*
 * Copyright 2017-2023 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.ml.ai.{MergeTokenStrategy, ZeroShotNerClassification}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForQuestionAnswering
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.scalatest.flatspec.AnyFlatSpec

class ZeroShotNerModelTest extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._

  "ZeroShotNerModel" should "load a RoBertaForQuestionAnswering instance via pretrained" taggedAs SlowTest in {
    ZeroShotNerModel
      .pretrained("roberta_base_qa_squad2", "en", "public/models")
      .isInstanceOf[ZeroShotNerModel]
  }

  "ZeroShotNer" should "download a RoBertaForQuestionAnswering and save it as a ZeroShotNerModel" taggedAs SlowTest in {

    RoBertaForQuestionAnswering
      .pretrained()
      .write
      .overwrite
      .save("./tmp_roberta_for_qa")

    val loadedZeroShotNerModel = ZeroShotNerModel
      .load("./tmp_roberta_for_qa")
      .setCaseSensitive(true)
      .setPredictionThreshold(0.1f)

    loadedZeroShotNerModel.write.overwrite
      .save("./tmp_roberta_for_qa_zero_ner")

  }

  "ZeroShotRobertaNer" should "run zero shot NER and check the number of entities returned" taggedAs SlowTest in {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = SentenceDetectorDLModel
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val zeroShotNer = ZeroShotNerModel
      .pretrained("roberta_base_qa_squad2")
      .setEntityDefinitions(
        Map(
          "NAME" -> Array("What is his name?", "What is my name?"),
          "CITY" -> Array("Which city?", "Which is the city?"),
          "SOMETHING_ELSE" -> Array("What is her name?")))
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("zero_shot_ner")
      .setIgnoreEntities(Array("SOMETHING_ELSE"))

    val nerConverter = new NerConverter()
      .setInputCols(Array("sentence", "token", "zero_shot_ner"))
      .setOutputCol("ner_chunks")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentenceDetector, tokenizer, zeroShotNer, nerConverter))

    val data = Seq(
      (
        "Hellen works in London, Paris and Berlin. My name is Clara Johnson, I live in New York and my sister Hellen lives in Paris.",
        6),
      ("John is a man who works in London, London and London.", 4)).toDF("text", "nEntities")

    val results = pipeline.fit(data).transform(data).cache()

    results
      .selectExpr("document", "explode(zero_shot_ner) AS entity")
      .select(
        col("document.result").getItem(0).alias("document"),
        col("entity.result"),
        col("entity.metadata.word"),
        col("entity.metadata.sentence"),
        col("entity.begin"),
        col("entity.end"),
        col("entity.metadata.confidence"),
        col("entity.metadata.question"))
      .show(truncate = false)

    results
      .selectExpr("size(ner_chunks)", "nEntities")
      .collect()
      .foreach(row =>
        assert(
          row.get(0).asInstanceOf[Int] == row.get(1).asInstanceOf[Int],
          s"expected ${row.get(1)} entities, got ${row.get(0)}"))

    results.select("zero_shot_ner.result").show(1, false)
    results.select("ner_chunks.result").show(1, false)

    println(zeroShotNer.getEntityDefinitionsStr.mkString("Array(", ", ", ")"))
    println(zeroShotNer.getIgnoreEntities.mkString("Array(", ", ", ")"))
    println(zeroShotNer.getEntities.mkString("Array(", ", ", ")"))
  }

  /** Regression test for SPARKNLP-45 (#14827): `RoBertaForQuestionAnswering.batchAnnotate`
    * switched from calling `predictSpan` once per row to `predictSpanGrouped`. `ZeroShotNerModel`
    * overrides `predictSpan`/`tagSpan` with its own decode (character offsets, not the base
    * class's token positions) but did not override `predictSpanGrouped`, so it silently fell
    * through to the generic RoBERTa QA decode the moment a real batch formed. Confirms the fixed
    * `predictSpanGrouped` matches the old per-row `predictSpan` exactly, on both compute engines
    * \- the fix also touches the ONNX attention mask, which the TensorFlow path alone can't
    * exercise.
    */
  "ZeroShotNerClassification" should "decode a batch exactly as it decodes one row at a time, on both engines" taggedAs SlowTest in {
    val text = "My name is Clara, I live in New York and Hellen lives in Paris."
    val questions = Seq(
      "What is his name?",
      "What is my name?",
      "What is her name?",
      "Which city?",
      "Which is the city?")

    def rows(context: Annotation): Seq[Seq[Annotation]] =
      questions.map(q =>
        Seq(
          Annotation(AnnotatorType.DOCUMENT, 0, q.length, q, Map.empty[String, String]),
          context))

    val contextAtZero =
      Annotation(AnnotatorType.DOCUMENT, 0, text.length - 1, text, Map("sentence" -> "0"))
    val prefix = "Nothing to see here. "
    val contextOffset = Annotation(
      AnnotatorType.DOCUMENT,
      prefix.length,
      prefix.length + text.length - 1,
      text,
      Map("sentence" -> "1"))

    // roberta_base_qa_squad2 and roberta_qa_distilroberta_base_squad_v2 are genuinely
    // ONNX-exported checkpoints (confirmed via detectedEngine below, not assumed from their
    // names - published checkpoints are sometimes re-exported to a different engine under the
    // same name over time). roberta_qa_robertaABSA is genuinely TensorFlow.
    Seq("tensorflow" -> "roberta_qa_robertaABSA", "onnx" -> "roberta_base_qa_squad2").foreach {
      case (engine, modelName) =>
        val qa = RoBertaForQuestionAnswering.pretrained(modelName, "en", "public/models")
        val inner = qa.getModelIfNotSet
        val model = new ZeroShotNerClassification(
          inner.tensorflowWrapper,
          inner.onnxWrapper,
          inner.openvinoWrapper,
          qa.sentenceStartTokenId,
          qa.sentenceEndTokenId,
          qa.padTokenId,
          false,
          configProtoBytes = qa.getConfigProtoBytes,
          tags = Map.empty[String, Int],
          signatures = qa.getSignatures,
          merges = qa.merges.get.get,
          vocabulary = qa.vocabulary.get.get)
        assert(
          model.detectedEngine == engine,
          "expected detectedEngine=" + engine + " but got " + model.detectedEngine)

        Seq(contextAtZero, contextOffset).foreach { context =>
          val reference = rows(context).map(row =>
            model
              .predictSpan(row, 512, caseSensitive = true, MergeTokenStrategy.vocab, engine)
              .head)

          Seq(1, 2, 3, 5, 8, 16).foreach { batchSize =>
            val batched = model
              .predictSpanGrouped(
                rows(context),
                batchSize,
                512,
                caseSensitive = true,
                MergeTokenStrategy.vocab,
                engine)
              .map(_.head)

            reference.zip(batched).zip(questions).foreach { case ((r, b), q) =>
              assert(
                r.result == b.result && r.begin == b.begin && r.end == b.end,
                "[" + engine + "] batchSize=" + batchSize + " question=[" + q + "]\n  expected: " + r + "\n  actual:   " + b)
            }
          }
        }
    }
  }
}
