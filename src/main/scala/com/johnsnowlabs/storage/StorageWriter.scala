package com.johnsnowlabs.storage

import org.rocksdb.{WriteBatch, WriteOptions}

abstract class StorageWriter[A](connection: RocksDBConnection,
                                autoFlushAfter: Option[Integer] = None
                               ) extends AutoCloseable {

  connection.connectReadWrite

  var batch = new WriteBatch()
  var batchSize = 0

  def flush(): Unit = {
    val writeOptions = new WriteOptions()
    connection.getDb.write(writeOptions, batch)
    batch.close()
    batch = new WriteBatch()
    batchSize = 0
  }

  def add(word: String, content: A): Unit = {
    batch.put(word.getBytes, toBytes(content))
    batchSize += 1

    if (autoFlushAfter.isDefined) {
      if (batchSize >= autoFlushAfter.get)
        flush()
    }
  }

  def toBytes(content: A): Array[Byte]

  override def close(): Unit = {
    if (batchSize > 0)
      flush()

    connection.close()
  }



}
