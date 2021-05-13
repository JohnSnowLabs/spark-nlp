package com.johnsnowlabs.ml.tensorflow

import org.junit.Assert.{assertArrayEquals, assertEquals}
import org.scalatest.FlatSpec
import org.tensorflow.Operand
import org.tensorflow.ndarray.{FloatNdArray, IntNdArray, StdArrays}
import org.tensorflow.op.core.Constant
import org.tensorflow.types.{TFloat32, TInt32}

class TensorResourcesTest extends FlatSpec with EagerSessionBuilder {

  "Tensor Resource" should "reverse a tensor" in {
    val matrix = StdArrays.ndCopyOf(Array[Array[Float]](Array(1.9f, 2.8f, 3.7f, 4.6f, 5.5f, 6.4f)))
    val tensor: Operand[TFloat32] = Constant.tensorOf(scope, matrix)
    val expectedReverseValues = Array[Float](6.4f, 5.5f, 4.6f, 3.7f, 2.8f, 1.9f)

    val reversedTensor = TensorResources.reverseTensor(scope, tensor, 1)

    val actualReverseValues = TensorResources.extractFloats(reversedTensor.asTensor())
    assertArrayEquals(expectedReverseValues, actualReverseValues, 0.1f)
  }

  it should "concat int tensors" in {
    val tensor1: Operand[TInt32] = Constant.tensorOf(scope, Array[Array[Int]](Array(1, 2, 3, 4, 5)))
    val tensor2: Operand[TInt32] = Constant.tensorOf(scope, Array[Array[Int]](Array(6, 7, 8, 9, 10)))
    val tensors = Array(tensor1, tensor2)
    val expectedConcat: IntNdArray = StdArrays.ndCopyOf(Array[Array[Int]](Array(1, 2, 3, 4, 5), Array(6, 7, 8, 9, 10)))

    val actualConcat = TensorResources.concatTensors(scope, tensors, 0)

    assertArrayEquals(Array[Long](2, 5), actualConcat.asTensor().shape.asArray)
    assertEquals(expectedConcat, actualConcat.data)
  }

  it should "concat float tensors" in {
    val tensor1: Operand[TFloat32] = Constant.tensorOf(scope, Array[Array[Float]](Array(1f, 2f, 3f, 4f, 5f)))
    val tensor2: Operand[TFloat32] = Constant.tensorOf(scope, Array[Array[Float]](Array(6f, 7f, 8f, 9f, 10f)))
    val tensors = Array(tensor1, tensor2)
    val expectedConcat: FloatNdArray = StdArrays.ndCopyOf(Array[Array[Float]](Array(1f, 2f, 3f, 4f, 5f),
      Array(6f, 7f, 8f, 9f, 10f)))

    val actualConcat = TensorResources.concatTensors(scope, tensors, 0)

    assertArrayEquals(Array[Long](2, 5), actualConcat.asTensor().shape.asArray)
    assertEquals(expectedConcat, actualConcat.data)
  }

  it should "reshape tensor" in {
    val matrix = StdArrays.ndCopyOf(Array[Array[Float]](Array(1.0f, 2.0f, 3.0f, 4.0f),
      Array(4.0f, 5.0f, 6.0f, 7.0f), Array(8.0f, 9.0f, 10.0f, 11.0f)))
    val tensor = Constant.tensorOf(scope, matrix)
    val shape = Array(3, 1, 4)

    val reshapeTensor = TensorResources.reshapeTensor(scope, tensor, shape)

    assertArrayEquals(shape.map(_.toLong), reshapeTensor.asTensor().shape.asArray)
  }

  it should "stack tensors" in {
    val tensorX: Operand[TInt32] = Constant.vectorOf(scope, Array[Int](1, 4))
    val tensorY: Operand[TInt32] = Constant.vectorOf(scope, Array[Int](2, 5))
    val tensorZ: Operand[TInt32] = Constant.vectorOf(scope, Array[Int](3, 6))
    val expectedShape: Array[Long] = Array(3, 2)

    val actualStackedTensor = TensorResources.stackTensors(scope, Array(tensorX, tensorY, tensorZ))

    assertArrayEquals(expectedShape, actualStackedTensor.asTensor().shape().asArray())
  }

}
