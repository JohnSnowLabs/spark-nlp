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

package com.johnsnowlabs.ml.crf

import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

class PredictionSpec extends FlatSpec {

  "CrfModel" should "return correct prediction" taggedAs FastTest in {
    val dataset = TestDatasets.small
    val weights = Array.fill(8)(0.1f)

    val model = new LinearChainCrfModel(weights, dataset.metadata)
    val instance = dataset.instances.head._2
    val labels = model.predict(instance)

    assert(labels.labels == Seq(1, 2))
  }

  "CrfModel" should "return correct prediction with negative sums" taggedAs FastTest in {
    val dataset = TestDatasets.small
    val weights = Array.fill(8)(-0.1f)

    val model = new LinearChainCrfModel(weights, dataset.metadata)
    val instance = dataset.instances.head._2
    val labels = model.predict(instance)

    assert(labels.labels == Seq(0, 0))
  }
}

