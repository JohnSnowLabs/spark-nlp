package com.johnsnowlabs.ml.tensorflow

import org.junit.Assert.assertArrayEquals
import org.scalatest.FlatSpec
import org.tensorflow.Operand
import org.tensorflow.op.core.{Constant, OnesLike, Zeros}
import org.tensorflow.types.TInt32

class TensorMathResourcesTest extends FlatSpec with EagerSessionBuilder {

  "TensorMathResourcesTest" should "sum an array of tensors" in {
    val tensorA: Operand[TInt32] = Constant.tensorOf(scope, Array[Array[Int]](Array(3, 5), Array(4, 8)))
    val tensorB: Operand[TInt32] = Constant.tensorOf(scope, Array[Array[Int]](Array(1, 6), Array(2, 9)))
    val expectedShape = tensorA.asTensor().shape().asArray()
    val expectedSum = Array(7, 16, 10, 25)

    val actualSum = TensorMathResources.sumTensors(scope, Array(tensorA, tensorB, tensorA))

    assertArrayEquals(expectedShape, actualSum.asTensor().shape().asArray())
    assertArrayEquals(expectedSum, TensorResources.extractInts(actualSum.asTensor()))
  }

  it should "subtract tensors" in {
    val shape = Constant.vectorOf(scope, Array(2, 2))
    val tensorA = Zeros.create(scope, shape, TInt32.DTYPE)
    val tensorB = OnesLike.create(scope, tensorA)
    val expectedShape = tensorA.asTensor().shape().asArray()
    val expectedSub = Array(-1, -1, -1, -1)

    val actualSub = TensorMathResources.subtractTensors(scope, tensorA, tensorB)

    assertArrayEquals(expectedShape, actualSub.asTensor().shape().asArray())
    assertArrayEquals(expectedSub, TensorResources.extractInts(actualSub.asTensor()))
  }

}
