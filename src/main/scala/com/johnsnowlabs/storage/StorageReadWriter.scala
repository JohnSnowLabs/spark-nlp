package com.johnsnowlabs.storage

import org.rocksdb.WriteBatch

trait StorageReadWriter[A] extends StorageWriter[A] {

  this: StorageReader[A] =>

  def add(word: String, content: A): Unit = {
    lru.update(word, Some(content))
    if (lru.getSize >= cacheSize) {
      flush(new WriteBatch())
    }
  }

  override def lookup(index: String): Option[A] = {
    val result = lru.getOrElseUpdate(index, lookupByIndex(index))
    if (lru.getSize >= cacheSize) {
      flush(new WriteBatch())
    }
    result
  }

  override def flush(batch: WriteBatch): Unit = {
    lru.foreach{case (word, content) =>
      put(batch, word, content)
    }
    super.flush(batch)
    lru.clear()
    if (connection.isConnected)
      connection.reconnectReadWrite
  }

  override def close(): Unit = {
    flush(new WriteBatch())
    super.close()
  }

}
