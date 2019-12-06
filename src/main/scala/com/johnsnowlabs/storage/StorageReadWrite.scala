package com.johnsnowlabs.storage

import java.nio.{ByteBuffer, ByteOrder}

import spire.ClassTag


abstract class StorageReadWrite[A: ClassTag] {

  protected val emptyValue: A

  protected def addToBuffer(buffer: ByteBuffer, content: A): Unit // buffer.putFloat(...)

  protected def getFromBuffer(buffer: ByteBuffer, index: Int): A // wrapper.getFloat(...)

  def toBytes(content: Array[A]): Array[Byte] = {
    val buffer = ByteBuffer.allocate(content.length * 4)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    for (value <- content) {
      addToBuffer(buffer, value)
    }
    buffer.array()
  }

  def fromBytes(source: Array[Byte]): Array[A] = {
    val wrapper = ByteBuffer.wrap(source)
    wrapper.order(ByteOrder.LITTLE_ENDIAN)
    val result = Array.fill[A](source.length / 4)(emptyValue)

    for (i <- result.indices) {
      result(i) = getFromBuffer(wrapper, i * 4)
    }
    result
  }
}