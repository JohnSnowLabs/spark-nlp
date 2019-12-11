package com.johnsnowlabs.storage

import java.nio.{ByteBuffer, ByteOrder}

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

  def add(word: String, content: Array[A]): Unit = {
    batch.put(word.getBytes, toBytes(content))
    batchSize += 1

    if (autoFlushAfter.isDefined) {
      if (batchSize >= autoFlushAfter.get)
        flush()
    }
  }

  protected def addToBuffer(buffer: ByteBuffer, content: A): Unit // buffer.putFloat(...)

  def toBytes(content: Array[A]): Array[Byte] = {
    val buffer = ByteBuffer.allocate(content.length * 4)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    for (value <- content) {
      addToBuffer(buffer, value)
    }
    buffer.array()
  }

  override def close(): Unit = {
    if (batchSize > 0)
      flush()

    connection.close()
  }



}
