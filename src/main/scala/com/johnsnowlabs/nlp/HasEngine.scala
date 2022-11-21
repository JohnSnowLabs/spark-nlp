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

import com.johnsnowlabs.ml.util.ModelEngine
import org.apache.spark.ml.param.Param

trait HasEngine extends ParamsAndFeaturesWritable {

  /** This param is set internally once via loadSavedModel. That's why there is no setter
    *
    * @group param
    */
  val engine = new Param[String](this, "engine", "Deep Learning engine used for this model")

  setDefault(engine, ModelEngine.tensorflow)

  /** @group getParam */
  def getEngine: String = $(engine)

}
