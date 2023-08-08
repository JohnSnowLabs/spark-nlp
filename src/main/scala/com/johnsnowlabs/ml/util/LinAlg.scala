package com.johnsnowlabs.ml.util

import breeze.linalg.{DenseMatrix, tile}

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

}
