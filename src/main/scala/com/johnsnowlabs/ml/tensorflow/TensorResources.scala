package com.johnsnowlabs.ml.tensorflow

import java.nio.{FloatBuffer, IntBuffer, LongBuffer}

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

  def calculateTensorSize(source: Tensor[_], size: Option[Int]): Int = {
    size.getOrElse{
      // Calculate real size from tensor shape
      val shape = source.shape()
      shape.foldLeft(1l)(_*_).toInt
    }
  }

  def extractInts(source: Tensor[_], size: Option[Int] = None): Array[Int] = {
    val realSize = calculateTensorSize(source ,size)
    val buffer = IntBuffer.allocate(realSize)
    source.writeTo(buffer)
    buffer.array()
  }

  def extractLongs(source: Tensor[_], size: Option[Int] = None): Array[Long] = {
    val realSize = calculateTensorSize(source ,size)
    val buffer = LongBuffer.allocate(realSize)
    source.writeTo(buffer)
    buffer.array()
  }

  def extractFloats(source: Tensor[_], size: Option[Int] = None): Array[Float] = {
    val realSize = calculateTensorSize(source ,size)
    val buffer = FloatBuffer.allocate(realSize)
    source.writeTo(buffer)
    buffer.array().map(item => item)
  }
}
