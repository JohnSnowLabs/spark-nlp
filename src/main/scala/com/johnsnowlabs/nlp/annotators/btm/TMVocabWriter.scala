package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.storage.{RocksDBConnection, StorageWriter}

class TMVocabWriter(connection: RocksDBConnection, autoFlushAfter: Option[Int])
  extends StorageWriter[BigInt](connection, autoFlushAfter) {

  override def toBytes(content: BigInt): Array[Byte] = {
    content.toByteArray
  }

}
