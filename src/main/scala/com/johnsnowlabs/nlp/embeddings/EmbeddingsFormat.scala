package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.storage.StorageFormat

object EmbeddingsFormat extends StorageFormat {
  type EmbeddingsFormat = Value
  val TEXT = Value
  val BINARY = Value
}