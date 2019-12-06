package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.storage.RocksDBReader

class WordEmbeddingsStorageReader(override val fileName: String, override val caseSensitive: Boolean, autoFlashAfter: Option[Integer] = None)
  extends RocksDBReader[Float](fileName, caseSensitive) {

  lazy val indexer: WordEmbeddingsReadWrite = new WordEmbeddingsReadWrite(localPath, "r", autoFlashAfter)

  override def customizedLookup(index: String): Option[Array[Float]] = {
    lazy val resultLower = findLocalDb.get(index.trim.toLowerCase.getBytes())
    lazy val resultUpper = findLocalDb.get(index.trim.toUpperCase.getBytes())
    lazy val resultExact = findLocalDb.get(index.trim.getBytes())

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
