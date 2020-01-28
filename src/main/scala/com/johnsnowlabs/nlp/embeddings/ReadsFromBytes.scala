package com.johnsnowlabs.nlp.embeddings

import java.nio.{ByteBuffer, ByteOrder}

trait ReadsFromBytes {

  def fromBytes(source: Array[Byte]): Array[Float] = {
    val wrapper = ByteBuffer.wrap(source)
    wrapper.order(ByteOrder.LITTLE_ENDIAN)
    val result = Array.fill[Float](source.length / 4)(0f)

    for (i <- result.indices) {
      result(i) = wrapper.getFloat(i * 4)
    }
    result
  }

}
