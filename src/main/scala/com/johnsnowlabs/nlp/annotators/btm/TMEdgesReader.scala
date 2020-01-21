package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader}

class TMEdgesReader(
                     override protected val connection: RocksDBConnection,
                     override protected val caseSensitiveIndex: Boolean
                   ) extends StorageReader[Int] {

  override def emptyValue: Int = -1

  override def fromBytes(source: Array[Byte]): Int = {
    BigInt(source).toInt
  }

  def lookup(index: (Int, Int)): Option[Int] = {
    super.lookup(index.toString())
  }

  override protected def readCacheSize: Int = 50000

}
