package com.johnsnowlabs.nlp.annotators.btm

import java.io.{ByteArrayOutputStream, ObjectOutputStream}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageBatchWriter}

class TMNodesWriter(
                    override protected val connection: RocksDBConnection
                  ) extends StorageBatchWriter[TrieNode] {

  def toBytes(content: TrieNode): Array[Byte] = {
    val stream: ByteArrayOutputStream = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(stream)
    oos.writeObject(content)
    oos.close()
    stream.toByteArray
  }

  def add(word: Int, value: TrieNode): Unit = {
    super.add(word.toString, value)
  }

  override protected def writeBufferSize: Int = 10000
}
