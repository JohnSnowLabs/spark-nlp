package com.johnsnowlabs.storage

object Database extends Enumeration {
  type Name = Value
  val EMBEDDINGS: Value = Value

  val TMVOCAB, TMEDGES, TMNODES = Value
}
