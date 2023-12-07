package com.johnsnowlabs.ml.util

import breeze.linalg.{DenseMatrix, tile}
import scala.math.sqrt

object LinAlg {

  object implicits {

    implicit class ExtendedDenseMatrix(m: DenseMatrix[Double]) {

      def shape: (Int, Int) = (m.rows, m.cols)

      /** Broadcast a DenseMatrix to a target matrix. Uses the same logic as numpy broadcasting.
        *
        * @param targetMatrix
        *   Target matrix with desired shape
        * @return
        *   Matrix with the same shape as the target matrix
        */
      def broadcastTo(targetMatrix: DenseMatrix[Double]): DenseMatrix[Double] =
        broadcastTo((targetMatrix.rows, targetMatrix.cols))

      /** Broadcast a DenseMatrix to an explicit shape. Uses the same logic as numpy broadcasting.
        *
        * @param shape
        *   Target shape of the matrix
        * @return
        *   Matrix with target shape
        */
      def broadcastTo(shape: (Int, Int)): DenseMatrix[Double] = {
        val (targetRows, targetCols) = shape

        require(
          targetRows >= m.rows && targetCols >= targetCols,
          "Can't broadcast to lower dimensions.")

        val sameRows = m.rows == targetRows
        val sameCols = m.cols == targetCols

        if (sameRows && sameCols)
          m
        else {
          // Same shape, or either one of them is 1
          val rowsCompatible: Boolean = sameRows || (m.rows == 1) || (targetRows == 1)
          val colsCompatible: Boolean = sameCols || (m.cols == 1) || (targetCols == 1)
          require(
            rowsCompatible && colsCompatible,
            s"Can't broadcast shape ${(m.rows, m.cols)} to $shape.")

          val tileRows = Math.max(targetRows - m.rows + 1, 1)
          val tileCols = Math.max(targetCols - m.cols + 1, 1)
          tile(m, tileRows, tileCols)
        }
      }
    }

  }

  def avgPooling(embeddings: Array[Float], attentionMask: Array[Long], dim: Int): Array[Float] = {
    val expandedAttentionMask = new Array[Float](embeddings.length)
    // Expand attentionMask to match the length of embeddings
    var j = 0
    for (i <- embeddings.indices) {
      expandedAttentionMask(i) = attentionMask(j)
      j += 1
      if (j == attentionMask.length) {
        j = 0 // reset j when we reach the end of attentionMask
      }
    }

    val sentenceEmbeddingsMatrix = embeddings.grouped(dim).toArray
    val attentionMaskMatrix = expandedAttentionMask.grouped(dim).toArray

    val elementWiseProduct =
      computeElementWiseProduct(sentenceEmbeddingsMatrix, attentionMaskMatrix)
    val weightedSum: Array[Float] = elementWiseProduct.transpose.map(_.sum)

    val sumAlongDimension2: Array[Float] = attentionMaskMatrix.transpose.map(_.sum)
    // Clamp each element to a minimum value of 1e-9
    val totalWeight: Array[Float] = sumAlongDimension2.map(x => math.max(x, 1e-9.toFloat))
    computeElementWiseDivision(weightedSum, totalWeight)
  }

  def computeElementWiseProduct(
      arrayA: Array[Array[Float]],
      arrayB: Array[Array[Float]]): Array[Array[Float]] = {
    arrayA.zip(arrayB).map { case (row1, row2) =>
      row1.zip(row2).map { case (a, b) => a * b }
    }
  }

  def computeElementWiseDivision(arrayA: Array[Float], arrayB: Array[Float]): Array[Float] = {
    arrayA.zip(arrayB).map { case (a, b) =>
      if (b != 0.0f) a / b else 0.0f // Avoid division by zero
    }
  }

  def normalizeArray(array: Array[Float]): Array[Float] = {
    val l2Norm: Float = sqrt(array.map(x => x * x).sum).toFloat
    // Normalize each element in the array
    array.map(value => if (l2Norm != 0.0f) value / l2Norm else 0.0f)
  }

}
