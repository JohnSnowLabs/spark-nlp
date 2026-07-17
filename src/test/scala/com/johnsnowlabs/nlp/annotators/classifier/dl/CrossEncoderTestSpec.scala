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

class CrossEncoderTestSpec extends AnyFlatSpec {

  import spark.implicits._

  lazy val document = new MultiDocumentAssembler()
    .setInputCols("query", "passage")
    .setOutputCols("document1", "document2")

  lazy val crossEncoder =
    CrossEncoder
      .pretrained()
      .setInputCols(Array("document1", "document2"))
      .setOutputCol("score")
      .setBatchSize(2)

  lazy val query = "How many people live in Berlin?"
  lazy val passages: Seq[String] = Seq(
    "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin is well known for its museums.",
    "In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.")

  lazy val data = passages.map(p => (query, p)).toDF("query", "passage")

  lazy val pipeline = new Pipeline().setStages(Array(document, crossEncoder))

  private def collectScores(df: org.apache.spark.sql.DataFrame): Seq[Float] =
    Annotation
      .collect(df, "score")
      .map(row => {
        assert(row.length == 1, "Exactly one score annotation per row")
        row.head.result.toFloat
      })

  behavior of "CrossEncoder"

  it should "produce exactly one sigmoid score per row" taggedAs SlowTest in {
    val pipelineDF = pipeline.fit(data).transform(data)
    pipelineDF.select("score.result", "score.metadata").show(false)

    val scores = collectScores(pipelineDF)
    assert(scores.length == passages.length, "One output row per input row expected")
    scores.foreach(s => assert(s >= 0.0f && s <= 1.0f, s"Score $s must be in [0, 1]"))
  }

  it should "rank the most relevant passage highest" taggedAs SlowTest in {
    val pipelineDF = pipeline.fit(data).transform(data)
    val scores = collectScores(pipelineDF)
    assert(scores.head == scores.max, "Most relevant passage should score highest")
  }

  it should "be serializable" taggedAs SlowTest in {
    val pipelineModel = pipeline.fit(data)
    pipelineModel.stages.last
      .asInstanceOf[CrossEncoder]
      .write
      .overwrite()
      .save("./tmp_cross_encoder_seq_classification")

    val loadedModel =
      CrossEncoder.load("./tmp_cross_encoder_seq_classification")
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
