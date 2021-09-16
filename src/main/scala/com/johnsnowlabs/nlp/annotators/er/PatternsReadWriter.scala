package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReadWriter}

class PatternsReadWriter(protected override val connection: RocksDBConnection)
  extends PatternsReader(connection) with StorageReadWriter[String] {

  override protected def writeBufferSize: Int = 10000

  override def toBytes(content: String): Array[Byte] = {
    content.getBytes()
  }

}
