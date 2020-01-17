package com.johnsnowlabs.nlp.embeddings

import java.nio.{ByteBuffer, ByteOrder}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReadWriter}

class WordEmbeddingsReadWriter(
                            override val connection: RocksDBConnection,
                            caseSensitiveIndex: Boolean,
                            dimension: Int,
                            maxCacheSize: Int
                          )
  extends WordEmbeddingsReader(connection, caseSensitiveIndex, dimension, maxCacheSize) with StorageReadWriter[Array[Float]] {

  override protected def writeBufferSize: Int = 1000

  override def toBytes(content: Array[Float]): Array[Byte] = {
    val buffer = ByteBuffer.allocate(content.length * 4)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    for (value <- content) {
      buffer.putFloat(value)
    }
    buffer.array()
  }

}