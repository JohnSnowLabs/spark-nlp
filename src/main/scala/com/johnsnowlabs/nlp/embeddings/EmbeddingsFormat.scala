package com.johnsnowlabs.nlp.embeddings

object EmbeddingsFormat extends Enumeration {
  type Format = Value
  val SPARKNLP = Value(1)
  val TEXT = Value(2)
  val BINARY = Value(3)
}