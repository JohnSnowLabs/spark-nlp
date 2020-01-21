package com.johnsnowlabs.nlp.embeddings

import java.nio.{ByteBuffer, ByteOrder}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader}

class WordEmbeddingsReader(
                            override val connection: RocksDBConnection,
                            override val caseSensitiveIndex: Boolean,
                            dimension: Int,
                            maxCacheSize: Int
                          )
  extends StorageReader[Array[Float]] {

  override def emptyValue: Array[Float] = Array.fill[Float](dimension)(0f)

  override def fromBytes(source: Array[Byte]): Array[Float] = {
    val wrapper = ByteBuffer.wrap(source)
    wrapper.order(ByteOrder.LITTLE_ENDIAN)
    val result = Array.fill[Float](source.length / 4)(0f)

    for (i <- result.indices) {
      result(i) = wrapper.getFloat(i * 4)
    }
    result
  }

  override protected def readCacheSize: Int = maxCacheSize
}