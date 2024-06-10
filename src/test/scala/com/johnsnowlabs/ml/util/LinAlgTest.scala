package com.johnsnowlabs.ml.util

import breeze.linalg._
import com.johnsnowlabs.ml.util.LinAlg.implicits.ExtendedDenseMatrix
import com.johnsnowlabs.ml.util.LinAlg.{denseMatrixToArray, l2Normalize}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LinAlgTest extends AnyFlatSpec with Matchers {

  behavior of "LinAlgTest"

  it should "broadcastTo" in {

    val targetShape = (3, 3)
    val shapes = Seq((3, 3), (1, 1), (1, 3), (3, 1))

    shapes.foreach { case (rows, cols) =>
      val m = DenseMatrix.zeros[Double](rows, cols)
      val broadcastM = m.broadcastTo(targetShape)
      assert(broadcastM.shape == targetShape, s"Casting of ${(rows, cols)} incorrect.")
    }

    val illegalShapes = Seq((3, 5), (5, 3), (5, 5), (2, 2))

    illegalShapes.foreach { case (rows, cols) =>
      assertThrows[IllegalArgumentException] {
        val m = DenseMatrix.zeros[Double](rows, cols)
        m.broadcastTo(targetShape)
      }
    }

  }

  it should "broadcast compatible with Matrix operations" in {
    val a = DenseVector(2.0d, 4.0d, 6.0d, 8.0d)
    val aMatrix = a.toDenseMatrix

    val b = DenseVector(1.0d, 3.0d)
    val bMatrix = b.toDenseMatrix.t

    println(aMatrix.shape, bMatrix.shape)

    val aExpanded = aMatrix.broadcastTo((b.length, a.length))
    val bExpanded = bMatrix.broadcastTo((b.length, a.length))

    val expected: DenseMatrix[Double] =
      DenseMatrix(Seq(1.0, 3.0, 5.0, 7.0), Seq(-1.0, 1.0, 3.0, 5.0))
    assert(aExpanded - bExpanded == expected)
  }

  it should "broadcast compatible with Matrix operations 2" in {
    val a: DenseMatrix[Double] = DenseMatrix.ones[Double](4, 2)
    val b: DenseMatrix[Double] = DenseVector(1.0d, 2.0d).toDenseMatrix.broadcastTo((4, 2))

    val ab = a + b
    val expected = DenseMatrix(Seq(2.0d, 3.0d), Seq(2.0d, 3.0d), Seq(2.0d, 3.0d), Seq(2.0d, 3.0d))
    assert(ab == expected)
  }

  val tolerance = 1e-6f

  def assertEqualWithTolerance(actual: Array[Float], expected: Array[Float]): Unit = {
    assert(actual.length == expected.length, "Array lengths differ")
    for ((a, e) <- actual.zip(expected)) {
      assert(math.abs(a - e) <= tolerance, s"Expected $e, got $a within tolerance $tolerance")
    }
  }

  "l2Normalize" should "correctly normalize a regular matrix" in {
    val matrix = DenseMatrix((1.0f, 2.0f), (3.0f, 4.0f))
    val normalized = l2Normalize(matrix)
    assertEqualWithTolerance(
      normalized(*, ::).map(norm(_, 2)).toArray.map(_.toFloat),
      Array(1.0f, 1.0f))
  }

  it should "handle a single row matrix" in {
    val matrix = DenseMatrix((1.0f, 2.0f, 3.0f))
    val normalized = l2Normalize(matrix)
    assert(math.abs(norm(normalized.toDenseVector, 2) - 1.0f) <= tolerance)
  }

  it should "handle a single column matrix" in {
    val matrix = DenseMatrix(1.0f, 2.0f, 3.0f)
    val normalized = l2Normalize(matrix)
    assertEqualWithTolerance(
      normalized(*, ::).map(norm(_, 2)).toArray.map(_.toFloat),
      Array(1.0f, 1.0f, 1.0f))
  }

  it should "handle a matrix with zero elements" in {
    val matrix = DenseMatrix((0.0f, 0.0f), (0.0f, 0.0f))
    val normalized = l2Normalize(matrix)
    assert(normalized === matrix)
  }

  it should "normalize each row to unit length" in {
    val matrix = DenseMatrix((1.0f, 0.0f), (0.0f, 1.0f))
    val normalized = l2Normalize(matrix)
    assertEqualWithTolerance(
      normalized(*, ::).map(norm(_, 2)).toArray.map(_.toFloat),
      Array(1.0f, 1.0f))
  }

  it should "correctly normalize a matrix with negative values" in {
    val matrix = DenseMatrix((-1.0f, -2.0f), (3.0f, -4.0f))
    val normalized = l2Normalize(matrix)
    assertEqualWithTolerance(
      normalized(*, ::).map(norm(_, 2)).toArray.map(_.toFloat),
      Array(1.0f, 1.0f))
  }

  "denseMatrixToArray" should "correctly convert a regular matrix" in {
    val matrix = DenseMatrix((1.0f, 2.0f), (3.0f, 4.0f))
    val array = denseMatrixToArray(matrix)
    assert(array === Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
  }

  it should "handle a single row matrix" in {
    val matrix = DenseMatrix.create(1, 3, Array(1.0f, 2.0f, 3.0f))
    val array = denseMatrixToArray(matrix)
    assert(array === Array(Array(1.0f, 2.0f, 3.0f)))
  }

  it should "handle a single column matrix" in {
    val matrix = DenseMatrix.create(3, 1, Array(1.0f, 2.0f, 3.0f))
    val array = denseMatrixToArray(matrix)
    assert(array === Array(Array(1.0f), Array(2.0f), Array(3.0f)))
  }

  it should "handle an empty matrix" in {
    val matrix = DenseMatrix.zeros[Float](0, 0)
    val array = denseMatrixToArray(matrix)
    assert(array === Array[Array[Float]]())
  }

  it should "correctly convert a matrix with various values" in {
    val matrix = DenseMatrix((-1.0f, 0.0f), (3.0f, -4.0f))
    val array = denseMatrixToArray(matrix)
    assert(array === Array(Array(-1.0f, 0.0f), Array(3.0f, -4.0f)))
  }

  "tokenPooling" should "correctly pool tokens for one index" in {
    val tokens: Array[Float] = Array(1, 2, 3, 4, 1, 2, 3, 4)
    val shape = Array(2, 4, 1)
    val embeddings = tokens.grouped(shape(2)).toArray.grouped(shape(1)).toArray
    val pooled = LinAlg.tokenPooling(embeddings, index = 0)

    assert(pooled.flatten sameElements Array(1.0f, 1.0f))
  }

  it should "correctly pool tokens for one index per sequence" in {
    val tokens: Array[Float] = Array(1, 2, 3, 4, 1, 2, 3, 4)
    val shape = Array(2, 4, 1)
    val embeddings = tokens.grouped(shape(2)).toArray.grouped(shape(1)).toArray
    val pooled =
      LinAlg.tokenPooling(embeddings, indexes = Array(0, 3))

    assert(pooled.flatten sameElements Array(1.0f, 4.0f))
  }

  "maxPooling" should "correctly pool tokens for one index" in {
    val embeddings: Array[Array[Array[Float]]] =
      Array(
        Array(Array(1.0f, 1.0f), Array(2.0f, 2.0f)),
        Array(Array(3.0f, 3.0f), Array(4.0f, 4.0f)))
    val attentionMask = Array(Array(1L, 1L), Array(1L, 1L))
    val pooled: Array[Array[Float]] = LinAlg.maxPooling(embeddings, attentionMask)

    assert(pooled(0) sameElements Array(2.0f, 2.0f))
    assert(pooled(1) sameElements Array(4.0f, 4.0f))
  }

  it should "consider attention mask" in {
    val embeddings: Array[Array[Array[Float]]] =
      Array(
        Array(Array(1.0f, 1.0f), Array(2.0f, 2.0f)),
        Array(Array(3.0f, 3.0f), Array(4.0f, 4.0f)))
    val attentionMask = Array(Array(1L, 0L), Array(1L, 0L))
    val pooled: Array[Array[Float]] = LinAlg.maxPooling(embeddings, attentionMask)

    assert(pooled(0) sameElements Array(1.0f, 1.0f))
    assert(pooled(1) sameElements Array(3.0f, 3.0f))
  }

  "clsAvgPooling" should "correctly pool tokens" in {
    val embeddings: Array[Array[Array[Float]]] =
      Array(
        Array(Array(1.0f, 2.0f), Array(1.0f, 2.0f)),
        Array(Array(3.0f, 4.0f), Array(3.0f, 4.0f)))
    val attentionMask = Array(Array(1L, 1L), Array(1L, 1L))
    val pooled: Array[Array[Float]] =
      LinAlg.clsAvgPooling(embeddings, attentionMask)

    assert(pooled(0) sameElements Array(1f, 2f))
    assert(pooled(1) sameElements Array(3f, 4f))
  }

  "lastPooling" should "correctly pool tokens" in {
    val embeddings: Array[Array[Array[Float]]] =
      Array(
        Array(Array(1.0f, 1.0f), Array(2.0f, 2.0f)),
        Array(Array(3.0f, 3.0f), Array(4.0f, 4.0f)))
    val attentionMask = Array(Array(1L, 1L), Array(1L, 1L))
    val pooled: Array[Array[Float]] =
      LinAlg.lastPooling(embeddings, attentionMask)

    assert(pooled(0) sameElements Array(2f, 2f))
    assert(pooled(1) sameElements Array(4f, 4f))
  }

  it should "correctly pool with padded sequences" in {
    val embeddings: Array[Array[Array[Float]]] =
      Array(
        Array(Array(1.0f, 1.0f), Array(2.0f, 2.0f), Array(-1f, -1f)),
        Array(Array(3.0f, 3.0f), Array(4.0f, 4.0f), Array(4.0f, 4.0f)))
    val attentionMask = Array(Array(1L, 1L, 0L), Array(1L, 1L, 1L))
    val pooled: Array[Array[Float]] =
      LinAlg.lastPooling(embeddings, attentionMask)

    assert(pooled(0) sameElements Array(2f, 2f))
    assert(pooled(1) sameElements Array(4f, 4f))
  }

}
