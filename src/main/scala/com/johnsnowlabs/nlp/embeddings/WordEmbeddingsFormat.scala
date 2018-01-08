package com.johnsnowlabs.nlp.embeddings

object WordEmbeddingsFormat extends Enumeration {
  type Format = Value

  val SparkNlp = Value(1)
  val Text = Value(2)
  val Binary = Value(3)
}
