/*
 * Copyright 2017-2021 John Snow Labs
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

import com.johnsnowlabs.nlp.LightPipeline
import com.johnsnowlabs.nlp.annotators.NormalizerModel
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class NerDLReaderTestSpec extends AnyFlatSpec {

  "Tensorflow NerDLReader" should "correctly load and save a ner model" taggedAs SlowTest in {

    val model = NerDLModelPythonReader.read(
      "./source_model",
      100,
      ResourceHelper.spark
    )
    model.write.overwrite().save("./some_model")

    succeed
  }


  "NerDLModel" should "correctly read and use a tensorflow originated ner model" taggedAs SlowTest in {
    val spark = ResourceHelper.spark
    import spark.implicits._

    val bp = PretrainedPipeline("pipeline_basic")

    bp.model.stages(2).asInstanceOf[NormalizerModel]

    val ner = NerDLModel.load("./some_model").setInputCols("document", "normal").setOutputCol("ner")

    val np = new Pipeline().setStages(Array(bp.model, ner))

    val target = Array(
      "With regard to the patient's chronic obstructive pulmonary disease, the patient's respiratory status improved throughout the remainder of her hospital course.")

    val fit = np.fit(Seq.empty[String].toDF("text"))

    val r = new LightPipeline(fit)
      .annotate(target)

    println(r.map(_.filterKeys(k => k == "document" || k == "ner")).mkString(","))

    succeed

  }

}
