package com.jsl.ml.crf

import VectorMath._

// ToDo make serializable
class LinearChainCrfModel(val weights: Array[Float], val metadata: DatasetMetadata) {
  val edgeCalculator = new EdgeCalculator(metadata)
  val labels = metadata.label2Id.size

  def predict(instance: Instance): InstanceLabels = {
    require(instance.items.size > 0)

    var newBestPath = Vector(labels)
    var bestPath = Vector(labels)

    val matrix = Matrix(labels, labels)
    edgeCalculator.fillLogEdges(instance.items.head.values, weights, 1f, matrix)
    copy(matrix(0), bestPath)

    val length = instance.items.length

    val prevIdx = Array.fill[Int](length, labels)(0)

    // Calculate best path
    for (i <- 1 until length) {
      val features = instance.items(i).values
      edgeCalculator.fillLogEdges(features, weights, 1f, matrix)
      fillVector(newBestPath, Float.MinValue)

      for (from <- 0 until labels) {
        for (to <- 0 until labels) {
          val newPath = bestPath(from) + matrix(from)(to)

          if (newBestPath(to) < newPath) {
            newBestPath(to) = newPath
            prevIdx(i)(to) = from
          }
        }
      }

      val tmp = newBestPath
      newBestPath = bestPath
      bestPath = tmp
    }

    // Restore best path
    val result = Array.fill(length)(0)
    var best = 0f
    for (i <- 0 until labels) {
      if (bestPath(i) > best) {
        best = bestPath(i)
        result(length - 1) = i
      }
    }

    for (i <- Range.inclusive(length - 2, 0, -1)) {
      result(i) = prevIdx(i + 1)(result(i + 1))
    }

    new InstanceLabels(result)
  }
}
