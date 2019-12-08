package com.johnsnowlabs.nlp.embeddings

import java.nio.ByteBuffer

import com.johnsnowlabs.storage.{RocksDBConnection, RocksDBWriter}

class WordEmbeddingsWriter(
                            connection: RocksDBConnection,
                            autoFlushAfter: Option[Integer] = None
                          )
  extends RocksDBWriter[Float](connection, autoFlushAfter) {

  override protected def addToBuffer(buffer: ByteBuffer, content: Float): Unit = {
    buffer.putFloat(content)
  }

}