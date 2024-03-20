package com.johnsnowlabs.ml.util

import breeze.linalg.{*, DenseMatrix, DenseVector, max, norm, sum, tile}

import scala.math.pow

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

  /** Calculates softmax probabilities for an array of logits.
    *
    * @param logitValues
    *   Predicted raw logits
    * @return
    *   Probabilities for each class
    */
  def softmax(logitValues: Array[Float]): Array[Float] = {
    val maxLogit = logitValues.max
    val logitsExp = logitValues.map(l => Math.exp(l - maxLogit))
    val expSum = logitsExp.sum
    logitsExp.map(exp => (exp / expSum).toFloat)
  }

  /** Gets the index with the highest score.
    *
    * @param scores
    *   Array of Scores to max
    * @return
    *   Index of the highest score
    */
  def argmax(scores: Array[Float]): Int =
    scores.zipWithIndex.maxBy { case (score, _) =>
      score
    }._2

  /** Performs average pooling on embeddings using an attention mask.
    *
    * This method takes flattened embeddings, an attention mask, and the shape of the embeddings,
    * and computes the average pooling. The pooling is done by grouping the embeddings based on
    * the attention mask and computing the weighted sum of these groups. The result is normalized
    * by the total weight of the attention mask.
    *
    * @param flattenEmbeddings
    *   Array of flattened embeddings
    * @param attentionMask
    *   2D Array representing the attention mask
    * @param shape
    *   Array representing the shape of the embeddings (dimensions)
    * @return
    *   A DenseMatrix of floats representing the average pooled embeddings
    */
  def avgPooling(
      flattenEmbeddings: Array[Float],
      attentionMask: Array[Array[Long]],
      shape: Array[Long]): DenseMatrix[Float] = {

    val thirdDim = shape.last.toInt
    val secondDim = shape(1).toInt
    val embeddings = flattenEmbeddings.grouped(thirdDim).grouped(secondDim).toArray

    val embeddingsMatrix = embeddings.map(embedding => DenseMatrix(embedding: _*))
    val attentionMaskMatrix = DenseMatrix(attentionMask: _*)
    val expandedAttentionMask = expandAttentionMask(embeddingsMatrix, attentionMaskMatrix)
    val weightedSum = computeWeightSum(embeddingsMatrix, expandedAttentionMask)
    val totalWeight = computeTotalWeight(expandedAttentionMask)
    weightedSum /:/ totalWeight
  }

  /** Expands the attention mask to match the dimensions of the token embeddings.
    *
    * This method is responsible for aligning the attention mask with the embeddings. It
    * transposes the attention mask and then replicates its values to match the dimensionality of
    * the token embeddings. The expansion is done for each slice of the embeddings, ensuring that
    * the expanded mask has the same number of rows as the token embeddings and the same number of
    * columns as the embedding dimension.
    *
    * @param embeddings
    *   Array of DenseMatrix[Float] representing the token embeddings
    * @param attentionMask
    *   DenseMatrix[Long] representing the initial attention mask
    * @return
    *   Array of DenseMatrix[Float] where each matrix is the expanded attention mask aligned with
    *   the corresponding token embeddings
    */

  private def expandAttentionMask(
      embeddings: Array[DenseMatrix[Float]],
      attentionMask: DenseMatrix[Long]): Array[DenseMatrix[Float]] = {

    val transposedMask = attentionMask.t
    val expectedEmbeddingSize = transposedMask.rows
    embeddings.map { embedding =>
      require(
        embedding.rows == expectedEmbeddingSize,
        s"Embedding dimension mismatch: expected $expectedEmbeddingSize, but found ${embedding.rows}")

      val embeddingSize = embedding.cols
      val expandedMask = DenseMatrix.zeros[Float](transposedMask.rows, embeddingSize)
      for (i <- 0 until transposedMask.rows; j <- 0 until embeddingSize) {
        expandedMask(i, j) =
          transposedMask(i, 0) // Replicate the mask value across the embedding dimension
      }

      expandedMask
    }
  }

  /** Computes the weighted sum of embeddings based on an expanded input mask.
    *
    * This method applies a weight to each embedding using the corresponding expanded input mask.
    * The weights are applied through element-wise multiplication of each embedding with its
    * respective mask. After weighting, the method sums the embeddings across the sequence length
    * dimension. The result is a DenseMatrix representing the weighted sum of the embeddings for
    * each item in the batch.
    *
    * @param embeddings
    *   Array of DenseMatrix[Float] representing the embeddings for each item in the batch
    * @param inputMaskExpanded
    *   Array of DenseMatrix[Float] representing the expanded input masks, aligned with the
    *   embeddings
    * @return
    *   DenseMatrix[Float] where each row corresponds to the weighted sum of embeddings for an
    *   item in the batch
    */
  private def computeWeightSum(
      embeddings: Array[DenseMatrix[Float]],
      inputMaskExpanded: Array[DenseMatrix[Float]]): DenseMatrix[Float] = {
    val batchSize = embeddings.length
    val embeddingDim = if (batchSize > 0) embeddings.head.cols else 0
    val resultMatrix = DenseMatrix.zeros[Float](batchSize, embeddingDim)

    for (i <- embeddings.indices) {
      val weighted = embeddings(i) *:* inputMaskExpanded(i)
      resultMatrix(i, ::) := sum(weighted(::, *))
    }

    resultMatrix
  }

  /** Computes the total weight for each embedding in the batch based on the expanded input mask.
    *
    * This method calculates the sum of weights for each embedding slice across the sequence
    * length dimension using the expanded input mask. The result is a DenseMatrix representing the
    * total weight for each embedding in the batch. To ensure numerical stability, a clamp
    * operation is applied to each sum to prevent values from falling below a minimum threshold.
    *
    * @param inputMaskExpanded
    *   Array of DenseMatrix[Float] representing the expanded input masks for each item in the
    *   batch
    * @param minValue
    *   Float representing the minimum value to clamp the weights to, defaulting to 1e-9f
    * @return
    *   DenseMatrix[Float] where each row corresponds to the total weight of embeddings for an
    *   item in the batch
    */
  private def computeTotalWeight(
      inputMaskExpanded: Array[DenseMatrix[Float]],
      minValue: Float = 1e-9f): DenseMatrix[Float] = {
    val batchSize = inputMaskExpanded.length
    val embeddingDim = if (batchSize > 0) inputMaskExpanded.head.cols else 0
    val totalWeight = DenseMatrix.zeros[Float](batchSize, embeddingDim)

    for (i <- inputMaskExpanded.indices) {
      totalWeight(i, ::) := sum(inputMaskExpanded(i)(::, *))
    }

    // Applying clamp operation
    totalWeight.mapValues(x => math.max(x, minValue))
  }

  /** Normalizes each row of a DenseMatrix using the L2 norm.
    *
    * This method applies L2 normalization to the embeddings. It first computes the L2 norm for
    * each row (embedding) in the input matrix. Then, it creates a matrix where each row is the
    * computed norms vector, ensuring the dimensions match the input embeddings. Finally, it
    * normalizes each row in the embeddings by dividing by the corresponding L2 norm.
    *
    * The result is a DenseMatrix where each row (embedding) is L2 normalized, ensuring that
    * embeddings have a consistent scale for further processing.
    *
    * @param embeddings
    *   DenseMatrix[Float] representing the embeddings to be normalized
    * @return
    *   DenseMatrix[Float] where each row is an L2 normalized version of the corresponding row in
    *   the input matrix
    */
  def l2Normalize(embeddings: DenseMatrix[Float]): DenseMatrix[Float] = {
    val norms = norm(embeddings(*, ::), 2)

    // Normalize each row, avoiding division by zero
    val normalized = DenseMatrix.tabulate[Float](embeddings.rows, embeddings.cols) { (i, j) =>
      if (norms(i) != 0) embeddings(i, j) / norms(i).toFloat else 0.0f
    }

    normalized
  }

  /** Converts a DenseMatrix to a 2D array of floats.
    *
    * This method is used to transform a DenseMatrix[Float] into a two-dimensional array. It
    * iterates over the rows and columns of the DenseMatrix, copying each element into the
    * corresponding position in the newly created 2D array.
    *
    * @param matrix
    *   DenseMatrix[Float] that needs to be converted to a 2D array
    * @return
    *   An 2D array representing the same data as the input DenseMatrix
    */
  def denseMatrixToArray(matrix: DenseMatrix[Float]): Array[Array[Float]] = {
    val rows = matrix.rows
    val cols = matrix.cols

    val array = Array.ofDim[Float](rows, cols)

    for (i <- 0 until rows) {
      for (j <- 0 until cols) {
        array(i)(j) = matrix(i, j)
      }
    }

    array
  }

  def lpNormalizeArray(array: Array[Float], p: Int = 2): Array[Float] = {
    val lpNorm: Float = pow(array.map(x => pow(x, p)).sum, 1.0 / p).toFloat
    // Normalize each element in the array
    array.map(value => if (lpNorm != 0.0f) value / lpNorm else 0.0f)
  }

  /** Creates pooled embeddings by selecting the token at the index position.
    *
    * @param embeddings
    *   Embeddings in shape (batchSize, sequenceLength, embeddingDim)
    * @param indexes
    *   Array of Index Positions to select for each sequence in the batch
    * @return
    *   A 2D array representing the pooled embeddings
    */
  def tokenPooling(
      embeddings: Array[Array[Array[Float]]],
      indexes: Array[Int]): Array[Array[Float]] = {
    val batchSize = embeddings.length
    require(indexes.length == batchSize, "Indexes length should be equal to batch size")

    embeddings.zip(indexes).map { case (tokens: Array[Array[Float]], index: Int) =>
      tokens(index)
    }
  }

  /** Creates pooled embeddings by selecting the token at the index position.
    *
    * @param embeddings
    *   Embeddings in shape (batchSize, sequenceLength, embeddingDim)
    * @param index
    *   Index Position to select for each sequence in the batch
    * @return
    *   A 2D array representing the pooled embeddings
    */
  def tokenPooling(embeddings: Array[Array[Array[Float]]], index: Int): Array[Array[Float]] =
    tokenPooling(embeddings, Array.fill(embeddings.length)(index))

  /** Creates pooled embeddings by taking the maximum of the embedding features along the
    * sequence.
    *
    * @param embeddings
    *   Embeddings in shape (batchSize, sequenceLength, embeddingDim)
    * @return
    *   A 2D array representing the pooled embeddings
    */
  def maxPooling(
      embeddings: Array[Array[Array[Float]]],
      attentionMask: Array[Array[Long]]): Array[Array[Float]] = {
    val embeddingsMatrix = embeddings.map(embedding => DenseMatrix(embedding: _*))

    val maskedEmbeddings: Array[DenseMatrix[Float]] =
      embeddingsMatrix.zip(attentionMask).map {
        case (embedding: DenseMatrix[Float], mask: Array[Long]) =>
          val maskVector: DenseVector[Float] = new DenseVector(mask.map(_.toFloat))
          embedding(::, *) *:* maskVector
      }

    maskedEmbeddings.map { seqEmbeddings: DenseMatrix[Float] =>
      max(seqEmbeddings(::, *)).t.toArray
    }
  }

  /** Creates pooled embeddings by using the CLS token as the representative embedding of the
    * sequence.
    *
    * @param embeddings
    *   Embeddings in shape (batchSize, sequenceLength, embeddingDim)
    * @param attentionMask
    *   Attention mask in shape (batchSize, sequenceLength)
    * @return
    *   The pooled embeddings in shape (batchSize, embeddingDim)
    */
  def clsPooling(
      embeddings: Array[Array[Array[Float]]],
      attentionMask: Array[Array[Long]]): Array[Array[Float]] = {
    tokenPooling(embeddings, 0) // CLS embedding is at the front of each sequence
  }

  /** Creates pooled embeddings by averaging the embeddings of the CLS token and the average
    * embedding the sequence.
    *
    * @param embeddings
    *   Embeddings in shape (batchSize, sequenceLength, embeddingDim)
    * @param attentionMask
    *   Attention mask in shape (batchSize, sequenceLength)
    * @return
    *   The pooled embeddings in shape (batchSize, embeddingDim)
    */
  def clsAvgPooling(
      embeddings: Array[Array[Array[Float]]],
      attentionMask: Array[Array[Long]]): Array[Array[Float]] = {
    val clsEmbeddings = DenseMatrix(clsPooling(embeddings, attentionMask): _*)
    val shape: Array[Long] =
      Array(embeddings.length, embeddings.head.length, embeddings.head.head.length)

    val flatEmbeddings: Array[Float] = embeddings.flatten.flatten
    val meanEmbeddings = avgPooling(flatEmbeddings, attentionMask, shape)

    val clsAvgEmbeddings = (clsEmbeddings +:+ meanEmbeddings) / 2.0f
    clsAvgEmbeddings.t.toArray // Breeze uses column-major order
      .grouped(meanEmbeddings.cols)
      .toArray
  }

  /** Creates pooled embeddings by taking the last token embedding of the sequence. Assumes right
    * padding.
    *
    * @param embeddings
    *   Embeddings in shape (batchSize, sequenceLength, embeddingDim)
    * @param attentionMask
    *   Attention mask in shape (batchSize, sequenceLength)
    * @return
    *   The pooled embeddings in shape (batchSize, embeddingDim)
    */
  def lastPooling(
      embeddings: Array[Array[Array[Float]]],
      attentionMask: Array[Array[Long]]): Array[Array[Float]] = {
    val lastTokenIndexes = attentionMask.map(_.sum.toInt - 1)

    tokenPooling(embeddings, lastTokenIndexes)
  }
}
