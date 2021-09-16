package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader}

class PatternsReader(protected val connection: RocksDBConnection) extends StorageReader[String] {

  override protected val caseSensitiveIndex: Boolean = false //TODO: Verify the value of this attribute

  override protected def readCacheSize: Int = 50000

  override def emptyValue: String = ""

  override def fromBytes(source: Array[Byte]): String = {
    new String(source)
  }

}
