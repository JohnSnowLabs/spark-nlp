package com.johnsnowlabs.storage

import org.rocksdb._
import spire.ClassTag


abstract class RocksDBReadWrite[A: ClassTag](dbFile: String, mode: String, autoFlashAfter: Option[Integer])
  extends StorageReadWrite[A] with RocksDBConnection with AutoCloseable {

  mode match {
    case "r" => openReadOnly(dbFile)
    case "w" => open(dbFile)
    case _ => throw new IllegalArgumentException("Invalid RocksDB open mode. Must be 'r' or 'w'")
  }

  var batch = new WriteBatch()
  var batchSize = 0

  def flush(): Unit = {
    val writeOptions = new WriteOptions()
    getDb.write(writeOptions, batch)
    batch.close()
    batch = new WriteBatch()
    batchSize = 0
  }

  def add(word: String, content: Array[A]): Unit = {
    batch.put(word.getBytes, toBytes(content))
    batchSize += 1

    if (autoFlashAfter.isDefined) {
      if (batchSize >= autoFlashAfter.get)
        flush()
    }
  }

  override def close(): Unit = {
    if (batchSize > 0)
      flush()

    super.close()
  }
}
