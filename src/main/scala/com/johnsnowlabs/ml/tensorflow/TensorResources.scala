package com.johnsnowlabs.ml.tensorflow

import org.tensorflow.Tensor
import scala.collection.mutable.ArrayBuffer
import scala.language.existentials


class TensorResources {
  private val tensors = ArrayBuffer[Tensor[_]]()

  def createTensor[T](obj: T): Tensor[_] = {
    val result = if (obj.isInstanceOf[String]) {
      Tensor.create(obj.asInstanceOf[String].getBytes("UTF-8"), classOf[String])
    }
    else {
      Tensor.create(obj)
    }

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
