package com.johnsnowlabs.nlp.embeddings

import java.io._
import com.johnsnowlabs.nlp.util.LruMap
import org.rocksdb._


case class WordEmbeddings(dbFile: String,
                          nDims: Int,
                          lruCacheSize: Int = 100000) extends Closeable{
  RocksDB.loadLibrary()

  val db = RocksDB.openReadOnly(dbFile)

  val zeroArray = Array.fill[Float](nDims)(0f)

  val lru = new LruMap[String, Array[Float]](lruCacheSize)

  private def getEmbeddingsFromDb(word: String): Array[Float] = {
    val result = db.get(word.toLowerCase.trim.getBytes())
    if (result == null)
      zeroArray
    else
      WordEmbeddingsIndexer.fromBytes(result)
  }

  def getEmbeddings(word: String): Array[Float] = {
    lru.getOrElseUpdate(word, getEmbeddingsFromDb(word))
  }

  override def close(): Unit = {
    db.close()
  }
}
