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

import com.johnsnowlabs.nlp.annotator.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class MPNetEmbeddingsTestSpec extends AnyFlatSpec {

  "Mpnet Embeddings" should "correctly embed multiple sentences" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("This is an example sentence", "Each sentence is converted")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = MPNetEmbeddings
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("mpnet")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(ddd).transform(ddd)
    pipelineDF.select("mpnet.embeddings").show(truncate = false)

  }

  it should "work with sentences" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._
    val testData = "I really enjoy my job. This is amazing"
    val testDf = Seq(testData).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetectorDL = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "en")
      .setInputCols(Array("document"))
      .setOutputCol("sentences")

    val embeddings = MPNetEmbeddings
      .pretrained()
      .setInputCols("sentences")
      .setOutputCol("mpnet")

    val pipeline = new Pipeline().setStages(Array(document, sentenceDetectorDL, embeddings))

    val pipelineModel = pipeline.fit(testDf)
    val pipelineDF = pipelineModel.transform(testDf)

    pipelineDF.select("mpnet.embeddings").show()
  }

}
