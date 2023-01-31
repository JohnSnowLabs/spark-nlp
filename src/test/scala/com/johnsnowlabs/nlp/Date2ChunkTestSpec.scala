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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.MultiDateMatcher
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

class Date2ChunkTestSpec extends AnyFlatSpec {

  "Date2Chunk" should "correctly converts DATE to CHUNK type" taggedAs FastTest in {
    import ResourceHelper.spark.implicits._

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(
      Array(
        """Omicron is a new variant of COVID-19, which the World Health Organization designated a variant of concern on Nov. 26, 2021/26/11.""",
        """Neighbouring Austria has already locked down its population this week for at until 2021/10/12, becoming the first to reimpose such restrictions."""))

    val dateChunkAnswer = Array(
      Seq[Annotation](
        Annotation(AnnotatorType.CHUNK, 118, 121, "2021/01/01", Map("sentence" -> "0"))),
      Seq[Annotation](
        Annotation(AnnotatorType.CHUNK, 83, 86, "2021/01/01", Map("sentence" -> "0"))))

    val inputFormats = Array("yyyy", "yyyy/dd/MM", "MM/yyyy", "yyyy")
    val outputFormat = "yyyy/MM/dd"

    val date = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(1900)
      .setInputFormats(inputFormats)
      .setOutputFormat(outputFormat)

    val multiDate = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("multi_date")
      .setAnchorDateYear(1900)
      .setInputFormats(inputFormats)
      .setOutputFormat(outputFormat)

    val date2Chunk = new Date2Chunk()
      .setInputCols("date")
      .setOutputCol("date_chunk")

    val multiDate2Chunk = new Date2Chunk()
      .setInputCols("multi_date")
      .setOutputCol("multi_date_chunk")

    val pipeline = new Pipeline().setStages(Array(date, multiDate, date2Chunk, multiDate2Chunk))

    val pipelineModel = pipeline.fit(data)
    val lightPipeline = new LightPipeline(pipelineModel)

    lightPipeline.annotate("Hello from Spark NLP in 2023 !")

    val dateChunkResults = pipelineModel
      .transform(data)
      .select("date_chunk")
      .as[Seq[Annotation]]
      .collect()

//    for ((a, b) <- dateChunkResults.zip(dateChunkAnswer)) {
//      assert(a == b)
//    }

    pipelineModel
      .transform(data)
      .select("date_chunk")
      .show(false)
  }

}
