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

import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotators.Token2Chunk
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

class RawAndSimpleAnnotatorTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  val sampleDataset: DataFrame = Seq[(String, String)](
    ("Hello world, this is a sentence out of nowhere", "a sentence out"),
    ("Hey there, there is no chunk here", ""),
    ("Woah here, don't go so fast", "this is not there")).toDF("text", "target")

  "Doc2Chunk" should "be loaded as PipelineModel" taggedAs FastTest in {

    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("document")

    val doc2Chunk =
      new Doc2Chunk().setInputCols("document").setChunkCol("target").setOutputCol("chunk")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, doc2Chunk))

    val pipelineModel = pipeline.fit(sampleDataset)

    pipelineModel.write.overwrite().save("./tmp_saved_pipeline")

    val loaded_pipelineModel = PipelineModel.load("./tmp_saved_pipeline")
    loaded_pipelineModel.transform(sampleDataset).show(10, truncate = false)

  }

  "Token2Chunk" should "be loaded as PipelineModel" taggedAs FastTest in {

    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("document")

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val token2Chunk =
      new Token2Chunk().setInputCols("token").setOutputCol("chunk")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, token, token2Chunk))

    val pipelineModel = pipeline.fit(Seq.empty[String].toDF("text"))

    pipelineModel.write.overwrite().save("./tmp_saved_pipeline")

    val loaded_pipelineModel = PipelineModel.load("./tmp_saved_pipeline")
    loaded_pipelineModel.transform(sampleDataset).show(10, truncate = false)

  }

}
