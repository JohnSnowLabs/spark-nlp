package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReadWriter}

class TMEdgesReadWriter(
                         protected override val connection: RocksDBConnection,
                         protected override val caseSensitiveIndex: Boolean
                       ) extends TMEdgesReader(connection, caseSensitiveIndex) with StorageReadWriter[Int] {

  def add(word: (Int, Int), content: Int): Unit = super.add(word.toString(), content)

  override protected def writeBufferSize: Int = 10000

  override def toBytes(content: Int): Array[Byte] = {
    BigInt(content).toByteArray
  }

}
