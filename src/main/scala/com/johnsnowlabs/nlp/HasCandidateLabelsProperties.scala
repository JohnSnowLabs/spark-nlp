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

import org.apache.spark.ml.param.{StringArrayParam, IntParam}

trait HasCandidateLabelsProperties extends ParamsAndFeaturesWritable {

  /** Candidate labels for classification, you can set candidateLabels dynamically during the
    * runtime
    *
    * @group param
    */
  val candidateLabels: StringArrayParam = new StringArrayParam(
    this,
    "candidateLabels",
    "Candidate labels for classification, you can set candidateLabels dynamically during the runtime")

  /** @group getParam */
  def getCandidateLabels: Array[String] = $(candidateLabels)

  /** @group setParam */
  def setCandidateLabels(value: Array[String]): this.type = set(candidateLabels, value)

  /** @group param */
  val entailmentIdParam = new IntParam(this, "entailmentIdParam", "")

  /** @group param */
  val contradictionIdParam = new IntParam(this, "contradictionIdParam", "")

  setDefault(
    candidateLabels -> Array("urgent", "not_urgent"),
    contradictionIdParam -> 0,
    entailmentIdParam -> 2)
}
