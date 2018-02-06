package com.johnsnowlabs.nlp.embeddings

object WordEmbeddingsFormat extends Enumeration {
  type Format = Value

  implicit def str2frm(v: String): Format = v.toUpperCase match {
    case "SPARKNLP" => SPARKNLP
    case "TEXT" => TEXT
    case "BINARY" => BINARY
    case _ => throw new Exception("Unsupported word embeddings format")
  }

  val SPARKNLP = Value(1)
  val TEXT = Value(2)
  val BINARY = Value(3)
}
