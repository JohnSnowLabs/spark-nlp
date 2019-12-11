package com.johnsnowlabs.nlp.embeddings

import java.nio.ByteBuffer

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader, StorageWriter}

class WordEmbeddingsReader(
                            connection: RocksDBConnection,
                            caseSensitiveIndex: Boolean,
                            lruCacheSize: Int = 100000
                          )
  extends StorageReader[Float](connection, caseSensitiveIndex, lruCacheSize) {

  override protected val emptyValue: Float = 0f

  override protected def getFromBuffer(buffer: ByteBuffer, index: Int): Float = {
    buffer.getFloat(index)
  }

}