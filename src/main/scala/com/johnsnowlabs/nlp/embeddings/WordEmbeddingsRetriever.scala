package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.storage.RocksDBRetriever

class WordEmbeddingsRetriever(dbFile: String,
                              caseSensitive: Boolean,
                              lruCacheSize: Int = 100000,
                              autoFlashAfter: Option[Integer] = Some(1000)
                             ) extends RocksDBRetriever[Float](dbFile, caseSensitive, lruCacheSize) {

  lazy val indexer: WordEmbeddingsIndexer = WordEmbeddingsIndexer(dbFile, autoFlashAfter)

  override def customizedLookup(index: String): Option[Array[Float]] = {
    lazy val resultLower = db.get(index.trim.toLowerCase.getBytes())
    lazy val resultUpper = db.get(index.trim.toUpperCase.getBytes())
    lazy val resultExact = db.get(index.trim.getBytes())

    if (caseSensitive && resultExact != null)
      Some(indexer.fromBytes(resultExact))
    else if (resultLower != null)
      Some(indexer.fromBytes(resultLower))
    else if (resultExact != null)
      Some(indexer.fromBytes(resultExact))
    else if (resultUpper != null)
      Some(indexer.fromBytes(resultUpper))
    else
      None

  }

}
