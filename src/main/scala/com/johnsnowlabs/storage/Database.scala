package com.johnsnowlabs.storage

trait Database extends Serializable {
  val name: String
  override def toString: String = {
    name
  }
}
object Database {
  type Name = Database
  val EMBEDDINGS: Name = new Name {
    override val name: String = "EMBEDDINGS"
  }
  val TMVOCAB: Name = new Name {
    override val name: String = "TMVOCAB"
  }
  val TMEDGES: Name = new Name {
    override val name: String = "TMEDGES"
  }
  val TMNODES: Name = new Name {
    override val name: String = "TMNODES"
  }
}
