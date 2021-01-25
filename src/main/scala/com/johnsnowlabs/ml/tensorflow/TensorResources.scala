package com.johnsnowlabs.ml.tensorflow

import java.nio.{ByteBuffer, FloatBuffer, IntBuffer, LongBuffer}

import org.tensorflow.Tensor

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.language.existentials

/**
  * This class is being used to initialize Tensors of different types and shapes for Tensorflow operations
  */
class TensorResources {
  private val tensors = ArrayBuffer[Tensor[_]]()

  def createTensor[T](obj: T): Tensor[_] = {
    val result = obj match {
      case str: String =>
        Tensor.create(str.getBytes("UTF-8"), classOf[String])
      case _ =>
        Tensor.create(obj)
    }

    tensors.append(result)
    result
  }


  def createIntBufferTensor[T](shape: Array[Long], buf: IntBuffer): Tensor[_] = {

    val result = Tensor.create(shape, buf)

    tensors.append(result)
    result
  }

  def createLongBufferTensor[T](shape: Array[Long], buf: LongBuffer): Tensor[_] = {

    val result = Tensor.create(shape, buf)

    tensors.append(result)
    result
  }

  def createFloatBufferTensor[T](shape: Array[Long], buf: FloatBuffer): Tensor[_] = {

    val result = Tensor.create(shape, buf)

    tensors.append(result)
    result
  }

  def createBytesBufferTensor[T](shape: Array[Long], buf: ByteBuffer): Tensor[_] = {

    val result = Tensor.create(classOf[String], shape, buf)

    tensors.append(result)
    result
  }

  def clearTensors(): Unit = {
    for (tensor <- tensors) {
      tensor.close()
    }

    tensors.clear()
  }

  def clearSession(outs: mutable.Buffer[Tensor[_]]): Unit = {
    outs.foreach(_.close())
  }

  def createIntBuffer(dim: Int): IntBuffer = {
    IntBuffer.allocate(dim)
  }

  def createLongBuffer(dim: Int): LongBuffer = {
    LongBuffer.allocate(dim)
  }

  def createFloatBuffer(dim: Int): FloatBuffer = {
    FloatBuffer.allocate(dim)
  }
}

object TensorResources {

  def calculateTensorSize(source: Tensor[_], size: Option[Int]): Int = {
    size.getOrElse{
      // Calculate real size from tensor shape
      val shape = source.shape()
      shape.foldLeft(1L)(_*_).toInt
    }
  }

  def extractInts(source: Tensor[_], size: Option[Int] = None): Array[Int] = {
    val realSize = calculateTensorSize(source ,size)
    val buffer = IntBuffer.allocate(realSize)
    source.writeTo(buffer)
    buffer.array()
  }

  def extractInt(source: Tensor[_], size: Option[Int] = None): Int =
    extractInts(source).head

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
    buffer.array()
  }
}
