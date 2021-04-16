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
import com.johnsnowlabs.tags.SlowTest

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

import org.scalatest._

class NerCrfCustomCase extends FlatSpec {

  val spark: SparkSession = ResourceHelper.spark


  "NerCRF" should "train CoNLL dataset" taggedAs SlowTest in {


    val conll = CoNLL()
    val test_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val embeddings = WordEmbeddingsModel.pretrained()
      .setInputCols("token", "sentence")
      .setOutputCol("embeddings")

    val nerCrf = new NerCrfApproach()
      .setInputCols("pos", "token", "sentence", "embeddings")
      .setOutputCol("ner")
      .setLabelColumn("label")
      .setMaxEpochs(50)

    val pipeline = new Pipeline()
      .setStages(Array(
        embeddings,
        nerCrf
      ))

    val model = pipeline.fit(test_data)

    model.write.overwrite().save("./tmp_nercrfpipeline")
    model.stages(1).asInstanceOf[NerCrfModel].write.overwrite().save("./tmp_nercrfmodel")

  }

  "NerCRF" should "load saved model and predict" taggedAs SlowTest in {

    val conll = CoNLL()
    val test_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testb")

    val m = PipelineModel.load("./tmp_nercrfpipeline")
    m.stages(1).asInstanceOf[NerCrfModel].setIncludeConfidence(false)

    m.transform(test_data.limit(2)).select("ner").show(false)
  }

}
