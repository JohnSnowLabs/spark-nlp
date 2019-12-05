package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.storage.StorageConnection

class WordEmbeddingsStorageConnection(override val fileName: String, override val caseSensitive: Boolean)
  extends StorageConnection[Float, WordEmbeddingsRetriever](fileName, caseSensitive) {

  override protected def createRetriever(localPath: String, caseSensitive: Boolean): WordEmbeddingsRetriever = {
    new WordEmbeddingsRetriever(localPath, caseSensitive)
  }

}
