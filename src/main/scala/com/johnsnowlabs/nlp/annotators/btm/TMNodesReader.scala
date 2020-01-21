package com.johnsnowlabs.nlp.annotators.btm

import java.io.{ByteArrayInputStream, ObjectInputStream}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader}

class TMNodesReader(
                     override val connection: RocksDBConnection,
                     override protected val caseSensitiveIndex: Boolean
                  ) extends StorageReader[TrieNode] {

  override def emptyValue: TrieNode = TrieNode(0, isLeaf = true, 0, 0)

  def lookup(index: Int): TrieNode = {
    super.lookup(index.toString).get
  }

  override def fromBytes(bytes: Array[Byte]): TrieNode = {
    val ois = new ObjectInputStream(new ByteArrayInputStream(bytes))
    val value = ois.readObject.asInstanceOf[TrieNode]
    ois.close()
    value
  }

  override protected def readCacheSize: Int = 50000

}
