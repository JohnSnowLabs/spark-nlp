package com.johnsnowlabs.ml.crf

import VectorMath._
import com.johnsnowlabs.nlp.annotators.param.{SerializedAnnotatorComponent, WritableAnnotatorComponent}


class LinearChainCrfModel(val weights: Array[Float], val metadata: DatasetMetadata)
  extends WritableAnnotatorComponent {

  val labels = metadata.label2Id.size

  def predict(instance: Instance): InstanceLabels = {
    if (instance.items.isEmpty)
      return InstanceLabels(Seq.empty)

    var newBestPath = Vector(labels)
    var bestPath = Vector(labels)

    val matrix = Matrix(labels, labels)
    EdgeCalculator.fillLogEdges(instance.items.head.values, weights, 1f, metadata, matrix)
    copy(matrix(0), bestPath)

    val length = instance.items.length

    val prevIdx = Array.fill[Int](length, labels)(0)

    // Calculate best path
    for (i <- 1 until length) {
      val features = instance.items(i).values
      EdgeCalculator.fillLogEdges(features, weights, 1f, metadata, matrix)
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

    InstanceLabels(result)
  }

  override def serialize: SerializedLinearChainCrfModel = {
    SerializedLinearChainCrfModel(
      weights.toList,
      metadata.serialize.asInstanceOf[SerializedDatasetMetadata]
    )
  }

  /**
    * Removes features with weights less then minW
    * @param minW Minimum weight to keep features
    * @return Shrinked model
    */
  def shrink(minW: Float): LinearChainCrfModel = {
    val (filteredWeights, featureIds) = weights.zipWithIndex.filter(p => Math.abs(p._1) >= minW).unzip
    val filteredMeta = metadata.filterFeatures(featureIds)
    new LinearChainCrfModel(filteredWeights, filteredMeta)
  }

}

case class SerializedLinearChainCrfModel
(
  weights: Seq[Float],
  metadata: SerializedDatasetMetadata
)
  extends SerializedAnnotatorComponent[LinearChainCrfModel]
{
  override def deserialize: LinearChainCrfModel = {
    new LinearChainCrfModel(
      weights.toArray,
      metadata.deserialize
    )
  }
}

