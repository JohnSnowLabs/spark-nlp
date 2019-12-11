package com.johnsnowlabs.nlp.embeddings

import java.nio.ByteBuffer

import com.johnsnowlabs.storage.{RocksDBConnection, StorageWriter}

class WordEmbeddingsWriter(
                            connection: RocksDBConnection,
                            autoFlushAfter: Option[Integer] = None
                          )
  extends StorageWriter[Float](connection, autoFlushAfter) {

  override protected def addToBuffer(buffer: ByteBuffer, content: Float): Unit = {
    buffer.putFloat(content)
  }

}