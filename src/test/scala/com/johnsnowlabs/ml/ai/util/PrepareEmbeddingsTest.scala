package com.johnsnowlabs.ml.ai.util

import org.scalatest.flatspec.AnyFlatSpec
import PrepareEmbeddings._

class PrepareEmbeddingsTest extends AnyFlatSpec {

  /** Helper to compare nested float arrays with small tolerance */
  private def approximatelyEqual(
      a: Seq[Array[Array[Float]]],
      b: Seq[Array[Array[Float]]],
      eps: Float = 1e-6f): Boolean = {
    a.indices.forall { i =>
      a(i).indices.forall { j =>
        a(i)(j).indices.forall { k =>
          Math.abs(a(i)(j)(k) - b(i)(j)(k)) < eps
        }
      }
    }
  }

  def createTestBatch(
      batchSize: Int,
      sentenceLength: Int,
      dim: Int): (Seq[Array[Int]], Array[Float]) = {
    val batchTokens = Seq.fill(batchSize)(Array.range(0, sentenceLength))
    val embeddings = (0 until batchSize * sentenceLength * dim).map(_.toFloat).toArray
    (batchTokens, embeddings)
  }

  def shape(arr: Seq[Array[Array[Float]]]): (Int, Int, Int) = {
    val batchSize = arr.length
    val sentenceLength = if (batchSize > 0) arr.head.length else 0
    val dim = if (sentenceLength > 0) arr.head.head.length else 0
    (batchSize, sentenceLength, dim)
  }

  def assertShape(
      arr: Seq[Array[Array[Float]]],
      batchSize: Int,
      maxSentenceLength: Int,
      dim: Int): Unit = {
    val actualShape = shape(arr)
    val expectedShape = (batchSize, maxSentenceLength, dim)
    assert(actualShape == expectedShape, s"Expected shape $expectedShape but got $actualShape")
  }

  /** Old Reference Implementation */
  def prepareBatchWordEmbeddingsRef(
      batch: Seq[Array[Int]],
      embeddings: Array[Float],
      maxSentenceLength: Int,
      batchLength: Int): Seq[Array[Array[Float]]] = {
    val dim = embeddings.length / (batchLength * maxSentenceLength)
    // Collection creation takes lots of RAM
    val batchEmbeddings: Array[Array[Array[Float]]] =
      embeddings
        .grouped(dim)
        .toArray
        .grouped(maxSentenceLength)
        .toArray

    val emptyVector = Array.fill(dim)(0f)

    batch.zip(batchEmbeddings).map { case (ids, embeddings) =>
      if (ids.length > embeddings.length) {
        embeddings.take(embeddings.length - 1) ++
          Array.fill(embeddings.length - ids.length)(emptyVector) ++
          Array(embeddings.last)
      } else {
        embeddings
      }
    }
  }

  behavior of "PrepareEmbeddingsTest"

  it should "produce identical results to the original version for a simple batch" in {
    val batchSize = 8
    val maxSentenceLength = 512
    val dim = 768

    val (batch, embeddings) = createTestBatch(batchSize, maxSentenceLength, dim)

    val original = prepareBatchWordEmbeddingsRef(batch, embeddings, maxSentenceLength, batchSize)
    val improved =
      prepareBatchWordEmbeddings(batch, embeddings, maxSentenceLength, batchSize)

    assertShape(original, batchSize, maxSentenceLength, dim)
    assertShape(improved, batchSize, maxSentenceLength, dim)

    assert(approximatelyEqual(original, improved))
  }

  it should "handle single-batch input correctly" in {
    val batchSize = 1
    val maxSentenceLength = 2
    val dim = 768

    val (batch, embeddings) = createTestBatch(batchSize, maxSentenceLength, dim)

    val original = prepareBatchWordEmbeddings(batch, embeddings, maxSentenceLength, batchSize)
    val improved =
      prepareBatchWordEmbeddings(batch, embeddings, maxSentenceLength, batchSize)

    assertShape(original, batchSize, maxSentenceLength, dim)
    assertShape(improved, batchSize, maxSentenceLength, dim)
    assert(approximatelyEqual(original, improved))
  }

  it should "work when sentence length is 1" in {
    val batchSize = 2
    val maxSentenceLength = 1
    val dim = 768

    val (batch, embeddings) = createTestBatch(batchSize, maxSentenceLength, dim)

    val original = prepareBatchWordEmbeddingsRef(batch, embeddings, maxSentenceLength, batchSize)
    val improved =
      prepareBatchWordEmbeddings(batch, embeddings, maxSentenceLength, batchSize)

    assertShape(original, batchSize, maxSentenceLength, dim)
    assertShape(improved, batchSize, maxSentenceLength, dim)
    assert(approximatelyEqual(original, improved))
  }

  it should "produce empty vectors when ids exceed available embeddings" in {
    val batch = Seq(Array(1, 2, 3, 4)) // longer than maxSentenceLength
    val tokenLength = batch.head.length
    val maxSentenceLength = 3
    val batchLength = 1
    val embeddings =
      (0 until batchLength * maxSentenceLength * tokenLength).map(_.toFloat).toArray // 1 * 3 * 4

    val original =
      prepareBatchWordEmbeddingsRef(batch, embeddings, maxSentenceLength, batchLength)
    val improved =
      prepareBatchWordEmbeddings(batch, embeddings, maxSentenceLength, batchLength)

    assert(approximatelyEqual(original, improved))
  }

}
