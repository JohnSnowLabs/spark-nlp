package com.johnsnowlabs.nlp.embeddings

import java.nio.{ByteBuffer, ByteOrder}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader}

class WordEmbeddingsReader(
                            connection: RocksDBConnection,
                            caseSensitiveIndex: Boolean,
                            lruCacheSize: Int = 100000
                          )
  extends StorageReader[Array[Float]](connection, caseSensitiveIndex, lruCacheSize) {

  override def emptyValue(size: Int): Array[Float] = Array.fill[Float](size)(0f)

  override def fromBytes(source: Array[Byte]): Array[Float] = {
    val wrapper = ByteBuffer.wrap(source)
    wrapper.order(ByteOrder.LITTLE_ENDIAN)
    val result = Array.fill[Float](source.length / 4)(0f)

    for (i <- result.indices) {
      result(i) = wrapper.getFloat(i * 4)
    }
    result
  }
}