package com.johnsnowlabs.storage

import org.rocksdb.{WriteBatch, WriteOptions}

trait StorageWriter[A] extends HasConnection {

  final protected var updatesCount = 0

  def toBytes(content: A): Array[Byte]

  def add(word: String, content: A): Unit

  protected def getUpdatesCount: Int = updatesCount

  protected def put(batch: WriteBatch, word: String, content: A): Unit = {
    batch.put(word.trim.getBytes, toBytes(content))
    updatesCount += 1
  }

  def flush(batch: WriteBatch): Unit = {
    val writeOptions = new WriteOptions()
    /** Might have disconnected already */
    if (connection.isConnected) {
      connection.getDb.write(writeOptions, batch)
    }
    batch.close()
    updatesCount = 0
  }

  def close(): Unit

}
