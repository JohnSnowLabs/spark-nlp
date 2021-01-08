package com.johnsnowlabs.ml.tensorflow

import java.nio.{ByteBuffer, FloatBuffer, IntBuffer, LongBuffer}

import org.tensorflow.ndarray.buffer._
import org.tensorflow.ndarray.{NdArray, NdArrays, Shape, StdArrays}
import org.tensorflow.types.{TFloat16, TInt32, TInt64, TString}
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

  def createLongBufferTensor(shape: Array[Long], buf: LongDataBuffer): Tensor[TInt64] = {

    val result = TInt64.tensorOf(Shape.of(shape:_*), buf)
    tensors.append(result)
    result
  }

  def createFloatBufferTensor(shape: Array[Long], buf: FloatDataBuffer): Tensor[TFloat16] = {

    val result = TFloat16.tensorOf(Shape.of(shape:_*), buf)
    tensors.append(result)
    result
  }

  /* not used anywhere...
  def createBytesBufferTensor[T](shape: Array[Long], buf: ByteDataBuffer): Tensor[_] = {

    val result = TByte.tensorOf(Shape.of(shape:_*), buf)
    tensors.append(result)
    result
  }*/
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
  // TODO all these implementations are not tested

  def calculateTensorSize(source: Tensor[_], size: Option[Int]): Int = {
    size.getOrElse{
      // Calculate real size from tensor shape
      val shape = source.shape()
      shape.asArray.foldLeft(1L)(_*_).toInt
    }
  }

  def extractInts(source: Tensor[_], size: Option[Int] = None): Array[Int] = {
    val realSize = calculateTensorSize(source ,size)
    val buffer = Array.fill(realSize)(0)
    source.rawData.asInts.write(buffer)
    buffer
  }

  def extractInt(source: Tensor[_], size: Option[Int] = None): Int =
    extractInts(source).head

  def extractLongs(source: Tensor[_], size: Option[Int] = None): Array[Long] = {
    val realSize = calculateTensorSize(source ,size)
    val buffer = Array.fill(realSize)(0L)
    source.rawData.asLongs.write(buffer)
    buffer
  }

  def extractFloats(source: Tensor[_], size: Option[Int] = None): Array[Float] = {
    val realSize = calculateTensorSize(source ,size)
    val buffer = Array.fill(realSize)(0f)
    source.rawData.asFloats.write(buffer)
    buffer
  }
}
