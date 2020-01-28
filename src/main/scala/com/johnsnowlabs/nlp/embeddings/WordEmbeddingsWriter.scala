package com.johnsnowlabs.nlp.embeddings

import java.nio.{ByteBuffer, ByteOrder}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageBatchWriter}

class WordEmbeddingsWriter(
                                override val connection: RocksDBConnection,
                                caseSensitiveIndex: Boolean,
                                dimension: Int,
                                maxCacheSize: Int,
                                writeBuffer: Int
                          )
  extends StorageBatchWriter[Array[Float]] with ReadsFromBytes {

  override protected def writeBufferSize: Int = writeBuffer

  override def toBytes(content: Array[Float]): Array[Byte] = {
    val buffer = ByteBuffer.allocate(content.length * 4)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    for (value <- content) {
      buffer.putFloat(value)
    }
    buffer.array()
  }

}