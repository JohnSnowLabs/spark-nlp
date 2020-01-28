package com.johnsnowlabs.nlp.embeddings

import java.nio.{ByteBuffer, ByteOrder}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader}

class WordEmbeddingsReader(
                            override val connection: RocksDBConnection,
                            override val caseSensitiveIndex: Boolean,
                            dimension: Int,
                            maxCacheSize: Int
                          )
  extends StorageReader[Array[Float]] with ReadsFromBytes {

  override def emptyValue: Array[Float] = Array.fill[Float](dimension)(0f)

  override protected def readCacheSize: Int = maxCacheSize
}