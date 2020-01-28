package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader}

class TMVocabReader(protected val connection: RocksDBConnection, protected val caseSensitiveIndex: Boolean)
  extends StorageReader[Int] {

  override def emptyValue: Int = -1

  override def fromBytes(source: Array[Byte]): Int = {
    BigInt(source).toInt
  }

  override protected def readCacheSize: Int = 50000

}
