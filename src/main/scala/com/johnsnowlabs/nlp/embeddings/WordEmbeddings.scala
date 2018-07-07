package com.johnsnowlabs.nlp.embeddings

import java.io._
import com.johnsnowlabs.nlp.util.LruMap
import org.rocksdb._


case class WordEmbeddings(dbFile: String,
                          nDims: Int,
                          normalize: Boolean,
                          lruCacheSize: Int = 100000) extends Closeable{
  RocksDB.loadLibrary()

  val db = RocksDB.openReadOnly(dbFile)

  val zeroArray = Array.fill[Float](nDims)(0f)

  val lru = new LruMap[String, Array[Float]](lruCacheSize)

  private def getEmbeddingsFromDb(word: String): Array[Float] = {
    lazy val result = db.get(word.toLowerCase.trim.getBytes())
    lazy val resultnn = db.get(word.trim.getBytes())
    if (normalize && result != null)
      WordEmbeddingsIndexer.fromBytes(result)
    else if (resultnn != null)
      WordEmbeddingsIndexer.fromBytes(resultnn)
    else
      zeroArray
  }

  def getEmbeddings(word: String): Array[Float] = {
    synchronized {
      lru.getOrElseUpdate(word, getEmbeddingsFromDb(word))
    }
  }

  def contains(word: String) = {
    (normalize && db.get(word.toLowerCase.trim.getBytes()) != null) || db.get(word.trim.getBytes()) != null
  }

  override def close(): Unit = {
    db.close()
  }
}
