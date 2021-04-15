/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Finisher, LightPipeline}
import com.johnsnowlabs.tags.SlowTest

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

import org.scalatest._

class NerCrfCustomCase extends FlatSpec {

  val spark: SparkSession = ResourceHelper.spark


  "NerCRF" should "read low trained model" taggedAs SlowTest in {


    val conll = CoNLL()
    val test_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testb")

    val embeddings = WordEmbeddingsModel.pretrained()
      .setInputCols("token", "sentence")
      .setOutputCol("embeddings")

    val nerCrf = new NerCrfApproach()
      .setInputCols("pos", "token", "sentence", "embeddings")
      .setOutputCol("ner")
      .setLabelColumn("label")
      .setMaxEpochs(5)

    val finisher = new Finisher()
      .setInputCols("ner")

    val pipeline = new Pipeline()
      .setStages(Array(
        embeddings,
        nerCrf,
        finisher
      ))

    val model = pipeline.fit(test_data)

    model.write.overwrite().save("./crfnerconll")
    model.stages(4).asInstanceOf[NerCrfModel].write.overwrite().save("./crfnerconll-single")

  }

  "NerCRF" should "read and predict" taggedAs SlowTest in {
    val lp = new LightPipeline(PipelineModel.load("./crfnerconll"))

    println(lp.annotate(
      "Lung, right lower lobe, lobectomy: Grade 3"
    ))

  }

}
