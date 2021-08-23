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

import VectorMath._

// Class helps with Forward-Backward algorithm values precalculations
class FbCalculator(val maxLength: Int, val metadata: DatasetMetadata) {

  val labels = metadata.label2Id.size
  val logPhi = Array.fill(maxLength)(Matrix(labels, labels))
  val phi = Array.fill(maxLength)(Matrix(labels, labels))
  val alpha = Array.fill(maxLength)(Vector(labels))
  val beta = Array.fill(maxLength)(Vector(labels))
  val c = Array.fill(maxLength)(1f)


  def calculate(sentence: Instance, weights: Array[Float], scale: Float): Unit = {
    require(sentence.items.length <= maxLength)

    calcPhi(sentence, weights, scale)
    calcAlpha(sentence)
    calcBeta(sentence)
  }

  private def calcPhi(sentence: Instance, weights: Array[Float], scale: Float): Unit = {
    val length = sentence.items.length

    for (i <- 0 until length) {
      // 1. Calculate log Phi for each edge
      EdgeCalculator.fillLogEdges(sentence.items(i).values, weights, scale, metadata, logPhi(i))

      // 2. Calc exp for each matrix value
      copy(logPhi(i), phi(i))
      exp(phi(i))
    }
  }

  // ToDo Try Linear Algebra operations on top of Matrises and Vectors
  private def calcAlpha(sentence: Instance): Unit = {
    val length = sentence.items.length
    require(length <= phi.length)

    fillMatrix(alpha, 0f)
    fillVector(c, 1f)

    copy(phi(0)(0), alpha(0))
    c(0) = alpha(0).sum
    multiply(alpha(0), 1 / c(0))

    var prev = alpha(0)

    for (i <- 1 until length) {
      for (from <- 0 until labels) {
        for (to <- 0 until labels) {
          alpha(i)(to) += prev(from) * phi(i)(from)(to)
        }
      }

      c(i) = alpha(i).sum
      require(c(i) != 0f)

      multiply(alpha(i), 1 / c(i))

      prev = alpha(i)
    }
  }

  private def calcBeta(sentence: Instance): Unit = {
    val length = sentence.items.length
    require(length <= phi.length)

    fillMatrix(beta, 0f)
    fillVector(beta(length - 1), 1f / c(length - 1))
    var next = beta(length - 1)

    for (i <- Range.inclusive(length - 2, 0, -1)) {
      for (from <- 0 until labels) {
        for (to <- 0 until labels) {
          beta(i)(from) += phi(i + 1)(from)(to) * next(to)
        }
      }

      multiply(beta(i), 1 / c(i))
      next = beta(i)
    }
  }

  def addObservedExpectations(weights: Vector,
                              instance: Instance,
                              instanceLabels: InstanceLabels,
                              c: Float): Unit = {

    val length = instance.items.length

    for (i <- 0 until length) {
      val label = instanceLabels.labels(i)

      // Observed Features
      for ((attrId, value) <- instance.items(i).values) {
        metadata.attrFeatures2Id
          .get((attrId, label))
          .foreach(fId => weights(fId) += c * value)
      }

      // Transition Features
      val fromLabel = if (i > 0) instanceLabels.labels(i - 1) else 0
      val meta = Transition(fromLabel, label)
      metadata.transFeature2Id.get(meta).foreach { fid =>
        weights(fid) += c
      }

    }
  }

  def addModelExpectations(weights: Vector,
                           sentence: Instance,
                           const: Float): Unit = {

    val length = sentence.items.length

    // Update Observed
    for (i <- 0 until length) {
      for ((attrId, value) <- sentence.items(i).values) {
        for (feature <- metadata.attr2Features(attrId)) {
          weights(feature.id) += const * c(i) * alpha(i)(feature.label) * beta(i)(feature.label) * value
        }
      }
    }

    // Update Transitions
    for (i <- 1 until length) {
      for ((feature, fid) <- metadata.transFeature2Id) {
        val from = feature.stateFrom
        val to = feature.stateTo

        weights(fid) += const * alpha(i - 1)(from) * phi(i)(from)(to) * beta(i)(to)
      }
    }

    // Update Transition from Start
    for ((feature, fid) <- metadata.transFeature2Id; if (feature.stateFrom == 0)) {
      val to = feature.stateTo
      weights(fid) += const * phi(0)(0)(to) * beta(0)(to)
    }
  }
}

object EdgeCalculator
{
  def fillLogEdges(values: Seq[(Int, Float)],
                   weights: Array[Float],
                   scale: Float,
                   metadata: DatasetMetadata,
                   matrix: Matrix): Unit = {

    val labels = metadata.labels.size

    fillMatrix(matrix, 0f)

    for ((attrId, value) <- values) {
      for (from <- 0 until labels)
        for (feature <- metadata.attr2Features(attrId))
          matrix(from)(feature.label) += weights(feature.id) * value * scale
    }

    for ((feature, fid) <- metadata.transFeature2Id) {
      matrix(feature.stateFrom)(feature.stateTo) += weights(fid) * scale
    }
  }
}
