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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.{LightPipeline, MultiDocumentAssembler}
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class CrossEncoderForSequenceClassificationTestSpec extends AnyFlatSpec {

  import spark.implicits._

  lazy val document = new MultiDocumentAssembler()
    .setInputCols("query", "passage")
    .setOutputCols("document1", "document2")

  lazy val crossEncoder =
    CrossEncoderForSequenceClassification
      .pretrained()
      .setInputCols(Array("document1", "document2"))
      .setOutputCol("score")
      .setBatchSize(2)

  // One query duplicated against several passages: a reranking layout the user builds upstream.
  lazy val query = "How many people live in Berlin?"
  lazy val passages: Seq[String] = Seq(
    "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin is well known for its museums.",
    "In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.")

  lazy val data = passages.map(p => (query, p)).toDF("query", "passage")

  lazy val pipeline = new Pipeline().setStages(Array(document, crossEncoder))

  behavior of "CrossEncoderForSequenceClassification"

  it should "score a locally imported ONNX model with loadSavedModel" taggedAs SlowTest in {
    // Point CROSS_ENCODER_MODEL_PATH at a locally exported cross-encoder (e.g. the ONNX
    // ms-marco-MiniLM-L6-v2 folder). The test is skipped when the env var is not set.
    val modelPath = sys.env.getOrElse("CROSS_ENCODER_MODEL_PATH", "")
    if (modelPath.isEmpty) cancel("CROSS_ENCODER_MODEL_PATH not set")

    val loaded = CrossEncoderForSequenceClassification
      .loadSavedModel(modelPath, spark)
      .setInputCols("document1", "document2")
      .setOutputCol("score")
      .setBatchSize(2)

    // The config declares max_position_embeddings = 512.
    assert(loaded.getModelMaxLength == 512)
    assert(loaded.getMaxSentenceLength == 512)

    val localPipeline = new Pipeline().setStages(Array(document, loaded))
    val result = localPipeline.fit(data).transform(data)

    result.select("score.result", "score.metadata").show(false)

    val scores = Annotation
      .collect(result, "score")
      .map(row => {
        assert(row.length == 1, "Exactly one score annotation per row")
        row.head.metadata("score").toFloat
      })

    assert(scores.length == passages.length, "One output row per input row")
    // The first passage directly answers the query and should score highest.
    assert(scores.head == scores.max, "Most relevant passage should score highest")
  }

  it should "produce exactly one score per row" taggedAs SlowTest in {
    val pipelineModel = pipeline.fit(data)
    val pipelineDF = pipelineModel.transform(data)

    pipelineDF.select("score.result", "score.metadata").show(false)

    val scores = Annotation.collect(pipelineDF, "score")
    assert(scores.length == passages.length, "One output row per input row expected")
    scores.foreach(row => assert(row.length == 1, "Exactly one score annotation per row"))
  }

  it should "rank the most relevant passage highest" taggedAs SlowTest in {
    val pipelineModel = pipeline.fit(data)
    val pipelineDF = pipelineModel.transform(data)

    val scores = Annotation
      .collect(pipelineDF, "score")
      .map(_.head.metadata("score").toFloat)

    // The first passage directly answers the query and should score highest.
    assert(scores.head == scores.max, "Most relevant passage should score highest")
  }

  it should "cap maxSentenceLength at the model config value" in {
    // Pure parameter logic: no model download needed.
    val model = new CrossEncoderForSequenceClassification().setModelMaxLength(128)
    model.setMaxSentenceLength(128) // at the ceiling: allowed
    assert(model.getMaxSentenceLength == 128)
    assertThrows[IllegalArgumentException] {
      model.setMaxSentenceLength(129) // above the ceiling: rejected
    }
  }

  it should "be serializable" taggedAs SlowTest in {
    val pipelineModel = pipeline.fit(data)
    pipelineModel.stages.last
      .asInstanceOf[CrossEncoderForSequenceClassification]
      .write
      .overwrite()
      .save("./tmp_cross_encoder_seq_classification")

    val loadedModel =
      CrossEncoderForSequenceClassification.load("./tmp_cross_encoder_seq_classification")
    val newPipeline = new Pipeline().setStages(Array(document, loadedModel))

    newPipeline.fit(data).transform(data).select("score.result").show(false)
  }

  it should "be compatible with LightPipeline" taggedAs SlowTest in {
    val pipelineModel = pipeline.fit(data)
    val lightPipeline = new LightPipeline(pipelineModel)
    val result = lightPipeline.fullAnnotate(
      Array(query),
      Array("Berlin has a population of 3,520,031 registered inhabitants."))
    assert(result.head("score").nonEmpty)
  }

}
