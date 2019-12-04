package com.johnsnowlabs.storage

import org.rocksdb._
import spire.ClassTag


abstract class RocksDbIndexer[A: ClassTag](dbFile: String, autoFlashAfter: Option[Integer]) extends StorageIndexer[A] with AutoCloseable {
  val options = new Options()
  options.setCreateIfMissing(true)
  options.setCompressionType(CompressionType.NO_COMPRESSION)
  options.setWriteBufferSize(20 * 1 << 20)

  RocksDB.loadLibrary()
  val writeOptions = new WriteOptions()

  val db: RocksDB = RocksDB.open(options, dbFile)
  var batch = new WriteBatch()
  var batchSize = 0

  def flush(): Unit = {
    db.write(writeOptions, batch)
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

    db.close()
  }
}
