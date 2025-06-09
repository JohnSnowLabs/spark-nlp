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

import com.johnsnowlabs.nlp.{AssertAnnotations, ImageAssembler}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Encoder, Encoders}
import org.apache.spark.sql.functions.{col, lit, size}
import org.scalatest.flatspec.AnyFlatSpec
import com.johnsnowlabs.nlp.util.EmbeddingsDataFrameUtils.{emptyImageRow, imageSchema}

class E5VEmbeddingsTestSpec extends AnyFlatSpec {
  lazy val model = getE5VEmbeddingsPipelineModel

  val textPrompt =
    "<|start_header_id|>user<|end_header_id|>\n\n<sent>\\nSummary above sentence in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
  val imagePrompt =
    "<|start_header_id|>user<|end_header_id|>\n\n<image>\\nSummary above image in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"

  "E5V Embeddings" should "correctly embed sentences" taggedAs SlowTest in {
    val testDF = getTestDF
    val result = model.transform(testDF)

    result.select("e5v.embeddings").show(true)

  }

  private def getTestDF: DataFrame = {
    val imageFolder = "src/test/resources/image1/"
    val imageDF: DataFrame = ResourceHelper.spark.read
      .format("image")
      .option("dropInvalid", value = true)
      .load(imageFolder)

    val testDF: DataFrame = imageDF.withColumn("text", lit(imagePrompt))
    val textDesc = "A cat sitting in a box."

    // Create DataFrame with a single null image row
    val spark = ResourceHelper.spark
    val nullImageDF =
      spark.createDataFrame(spark.sparkContext.parallelize(Seq(emptyImageRow)), imageSchema)

    val textDF = nullImageDF.withColumn("text", lit(textPrompt.replace("<sent>", textDesc)))

    testDF.union(textDF)
//    textDF
  }
  private def getE5VEmbeddingsPipelineModel = {
    val testDF = getTestDF

    val imageAssembler: ImageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")

    val loadModel = E5VEmbeddings
      .pretrained()
      .setInputCols("image_assembler")
      .setOutputCol("e5v")

    val newPipeline: Pipeline =
      new Pipeline().setStages(Array(imageAssembler, loadModel))

    val pipelineModel = newPipeline.fit(testDF)

    pipelineModel
      .transform(testDF)
      .show(truncate = true)

    pipelineModel
  }
}
