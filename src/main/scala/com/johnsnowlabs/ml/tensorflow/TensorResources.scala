package com.johnsnowlabs.ml.tensorflow

import org.tensorflow.Tensor
import scala.collection.mutable.ArrayBuffer


class TensorResources {
  private val tensors = ArrayBuffer[Tensor[_]]()

  def createTensor[T](obj: T): Tensor[_] =  {
    val result = Tensor.create(obj)
    tensors.append(result)
    result
  }

  def clearTensors(): Unit = {
    for (tensor <- tensors) {
      tensor.close()
    }

    tensors.clear()
  }
}
