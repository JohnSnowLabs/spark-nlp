package com.johnsnowlabs.nlp.embeddings

import java.nio.{ByteBuffer, ByteOrder}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageWriter}

class WordEmbeddingsWriter(
                            connection: RocksDBConnection,
                            autoFlushAfter: Option[Integer] = None
                          )
  extends StorageWriter[Array[Float]](connection, autoFlushAfter) {

  override def toBytes(content: Array[Float]): Array[Byte] = {
    val buffer = ByteBuffer.allocate(content.length * 4)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    for (value <- content) {
      buffer.putFloat(value)
    }
    buffer.array()
  }
}