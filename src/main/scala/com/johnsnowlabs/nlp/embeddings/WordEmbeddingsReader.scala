package com.johnsnowlabs.nlp.embeddings

import java.nio.ByteBuffer

import com.johnsnowlabs.storage.{RocksDBConnection, RocksDBReader, RocksDBWriter}

class WordEmbeddingsReader(
                            connection: RocksDBConnection,
                            caseSensitiveIndex: Boolean,
                            lruCacheSize: Int = 100000
                          )
  extends RocksDBReader[Float](connection, caseSensitiveIndex, lruCacheSize) {

  override protected val emptyValue: Float = 0f

  override protected def getFromBuffer(buffer: ByteBuffer, index: Int): Float = {
    buffer.getFloat(index)
  }

}