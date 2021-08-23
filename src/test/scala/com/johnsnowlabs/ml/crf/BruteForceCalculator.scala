/*
 * Copyright 2017-2019 John Snow Labs
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

import scala.collection.mutable.ArrayBuffer


class BruteForceCalculator(val metadata: DatasetMetadata, val fbCalculator: FbCalculator) {

  val labels = metadata.label2Id.size

  def calcExpectation(instance: Instance, feature: AttrFeature, values: Seq[Float]): Float = {
    val length = instance.items.length
    val paths = generatePaths(length + 1, 0)
    val z = paths.map(path => getProbe(path)).sum
    var expectation = 0f

    for (i <- 0 until length) {
      expectation += paths
        .filter(path => path(i + 1) == feature.label)
        .map(path => getProbe(path) * values(i))
        .sum
    }

    expectation / z
  }

  def calcExpectation(instance: Instance, transition: Transition): Float = {
    val length = instance.items.length + 1
    val paths = generatePaths(length, 0)
    val z = paths.map(path => getProbe(path)).sum
    var expectation = 0f

    for (i <- 1 to instance.items.length) {
      expectation += paths
        .filter(path => path(i-1) == transition.stateFrom
          && path(i) == transition.stateTo)
        .map(path => getProbe(path))
        .sum
    }

    expectation / z
  }

  def getProbe(instance: Instance, idx: Int, label: Int): Float = {
    val length = instance.items.length
    val paths = generatePaths(length + 1, 0)
    val z = paths.map(path => getProbe(path)).sum

    val expectation = paths
      .filter(path => path(idx + 1) == label)
      .map(path => getProbe(path))
      .sum

    expectation / z
  }

  private def getProbe(path: List[Int]): Float = {
    var probe = 1f
    for (i <- 0 until path.length - 1) {
      probe *= fbCalculator.phi(i)(path(i))(path(i + 1))
    }

    probe
  }

  private def generatePaths(length: Int, start: Int): Seq[List[Int]] = {
    if (length == 1)
      List(start :: Nil)
    else {
      val result = ArrayBuffer[List[Int]]()
      for (to <- 0 until labels) {
        for (path <- generatePaths(length - 1, to))
          result.append(start :: path)
      }
      result
    }
  }
}
