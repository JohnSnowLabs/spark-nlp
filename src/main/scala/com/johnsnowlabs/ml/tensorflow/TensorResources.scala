package com.johnsnowlabs.ml.tensorflow

import java.nio.{FloatBuffer, LongBuffer}

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

object TensorResources {

  def extractInts(source: Tensor[_], size: Int): Array[Int] = {
    val buffer = LongBuffer.allocate(size)
    source.writeTo(buffer)
    buffer.array().map(item => item.toInt)
  }

  def extractFloats(source: Tensor[_], size: Int): Array[Float] = {
    val buffer = FloatBuffer.allocate(size)
    source.writeTo(buffer)
    buffer.array().map(item => item.toFloat)
  }
}
