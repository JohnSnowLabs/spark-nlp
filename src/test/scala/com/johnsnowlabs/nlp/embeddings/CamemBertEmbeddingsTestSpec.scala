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

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

import scala.language.postfixOps

class CamemBertEmbeddingsTestSpec extends AnyFlatSpec {

  "CamemBert Embeddings" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("Something is weird on the notebooks, something is happening.").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val tfModelPath = "PATH"

    val embeddings = CamemBertEmbeddings
      .loadSavedModel(tfModelPath, ResourceHelper.spark)
      .setInputCols(Array("token", "document"))
      .setOutputCol("bert")
      .setStorageRef("tf_hub_bert_test")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    pipeline.fit(ddd).write.overwrite().save("./tmp_camembert_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_camembert_pipeline")

    pipelineModel.transform(ddd)
  }
}
