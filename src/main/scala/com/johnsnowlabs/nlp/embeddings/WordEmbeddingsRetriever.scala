package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.util.LruMap
import org.rocksdb._


case class WordEmbeddingsRetriever(dbFile: String,
                                   nDims: Int,
                                   caseSensitive: Boolean,
                                   lruCacheSize: Int = 100000) extends AutoCloseable {

  @transient private var prefetchedDB: RocksDB = null

  private def db: RocksDB = {
    if (Option(prefetchedDB).isDefined)
      prefetchedDB
    else {
      RocksDB.loadLibrary()
      prefetchedDB = RocksDB.openReadOnly(dbFile)
      prefetchedDB
    }
  }

  val zeroArray: Array[Float] = Array.fill[Float](nDims)(0f)

  val lru = new LruMap[String, Option[Array[Float]]](lruCacheSize)

  private def getEmbeddingsFromDb(word: String): Option[Array[Float]] = {
    lazy val resultLower = db.get(word.trim.toLowerCase.getBytes())
    lazy val resultUpper = db.get(word.trim.toUpperCase.getBytes())
    lazy val resultExact = db.get(word.trim.getBytes())

    if (caseSensitive && resultExact != null)
      Some(WordEmbeddingsIndexer.fromBytes(resultExact))
    else if (resultLower != null)
      Some(WordEmbeddingsIndexer.fromBytes(resultLower))
    else if (resultExact != null)
      Some(WordEmbeddingsIndexer.fromBytes(resultExact))
    else if (resultUpper != null)
      Some(WordEmbeddingsIndexer.fromBytes(resultUpper))
    else
      None

  }

  def getEmbeddingsVector(word: String): Option[Array[Float]] = {
    synchronized {
      lru.getOrElseUpdate(word, getEmbeddingsFromDb(word))
    }
  }

  def containsEmbeddingsVector(word: String): Boolean = {
    val wordBytes = word.trim.getBytes()
    db.get(wordBytes) != null ||
      (db.get(word.trim.toLowerCase.getBytes()) != null) ||
      (db.get(word.trim.toUpperCase.getBytes()) != null)

  }

  override def close(): Unit = {
    if (Option(prefetchedDB).isDefined) {
      db.close()
      prefetchedDB = null
    }
  }
}
