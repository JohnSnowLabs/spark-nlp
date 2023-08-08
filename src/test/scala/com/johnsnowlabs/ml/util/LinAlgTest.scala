package com.johnsnowlabs.ml.util

import breeze.linalg._
import com.johnsnowlabs.ml.util.LinAlg.implicits.ExtendedDenseMatrix
import org.scalatest.flatspec.AnyFlatSpec
class LinAlgTest extends AnyFlatSpec {

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
}
