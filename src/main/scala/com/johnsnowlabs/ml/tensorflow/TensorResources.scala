package com.johnsnowlabs.ml.tensorflow

import java.nio.{ByteBuffer, FloatBuffer, IntBuffer, LongBuffer}

import org.tensorflow.ndarray.buffer.{DataBuffers, FloatDataBuffer, IntDataBuffer}
import org.tensorflow.ndarray.{NdArray, NdArrays, Shape, StdArrays}
import org.tensorflow.types.{TFloat16, TInt32, TString}
import org.tensorflow.{DataType, Tensor}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.language.existentials

/**
  * This class is beeing used to initialize Tensors of different types and shapes for Tensorflow operations
  */
class TensorResources {
  private val tensors = ArrayBuffer[Tensor[_]]()

  def createTensor[T](obj: T): Tensor[_] = {
    val result = obj match {
      case str: String =>
        TString.scalarOf(str)
      case bidimArray:Array[Array[Float]] =>
        val flatten = bidimArray.flatten
        // old-style factory method currently discouraged
        // TODO TFloat16 o 32?
        TFloat16.tensorOf(StdArrays.ndCopyOf(bidimArray))
    }
    tensors.append(result)
    result
  }


  def createIntBufferTensor(shape: Array[Long], buf: IntDataBuffer): Tensor[TInt32] = {

    val result = TInt32.tensorOf(Shape.of(shape:_*), buf)
    tensors.append(result)
    result
  }

  def createLongBufferTensor(shape: Array[Long], buf: LongBuffer): Tensor[_] = {

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
    buffer.array().map(item => item)
  }
}
