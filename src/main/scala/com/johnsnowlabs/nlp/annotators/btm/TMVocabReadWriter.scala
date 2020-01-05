package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReadWriter}

class TMVocabReadWriter(
                         protected override val connection: RocksDBConnection,
                         protected override val caseSensitiveIndex: Boolean
                       )
  extends TMVocabReader(connection, caseSensitiveIndex) with StorageReadWriter[Int] {

  override def toBytes(content: Int): Array[Byte] = {
    BigInt(content).toByteArray
  }

}
