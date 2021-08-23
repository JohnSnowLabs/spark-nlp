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

import com.johnsnowlabs.ml.crf.VectorMath._
import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

class GradientSpec extends FlatSpec {
  val dataset = TestDatasets.small
  val instance = dataset.instances.head._2
  val metadata = dataset.metadata
  val features = metadata.attrFeatures.size + metadata.transitions.size
  val weights = Vector(features, 0.1f)

  val fb = new FbCalculator(2, metadata)
  val bruteForce = new BruteForceCalculator(metadata, fb)
  fb.calculate(instance, weights, 1f)


  "SGD" should "correctly calculates data estimation" taggedAs FastTest in {
    val instance = dataset.instances.head._2
    val labels = dataset.instances.head._1

    val features = dataset.metadata.attrFeatures.size + dataset.metadata.transitions.size
    val weights = Vector(features, 0.1f)
    fb.addObservedExpectations(weights, instance, labels, 0.1f)

    assert(weights.toSeq == Seq(0.2f, 0.2f, 0.3f, 0.2f, 0.3f, 0.4f, 0.2f, 0.2f))
  }


  "SGD" should "correctly calculates model estimation" taggedAs FastTest in {

    // 1. Calculate Model Expectation by Test BruteForce Algo
    val attrExp = metadata.attrFeatures.map{f =>
      val featureValues = instance.items.map(word => word.apply(f.attrId))
      bruteForce.calcExpectation(instance, f, featureValues)
    }

    val transExp = metadata.transitions.map(t =>
      bruteForce.calcExpectation(instance, t)
    )

    val a = -0.1f
    val expectations = (attrExp ++ transExp).toList
    val newWeights = expectations.map(e => 0.1f + a * e)

    // 2. Calculate Model Expectation by CRF Algo
    val weights = this.weights.clone()
    fb.addModelExpectations(weights, instance, a)

    // 3. Weights must be equal
    FloatAssert.seqEquals(weights.toSeq, newWeights)
  }

}
