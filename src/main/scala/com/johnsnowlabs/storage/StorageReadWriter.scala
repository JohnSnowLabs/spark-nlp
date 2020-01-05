package com.johnsnowlabs.storage

import org.rocksdb.WriteBatch

trait StorageReadWriter[A] extends StorageWriter[A] {

  this: StorageReader[A] =>

  def add(word: String, content: A): Unit = {
    if (lru.getSize >= cacheSize) {
      flush(new WriteBatch())
    }
    lru.update(word, Some(content))
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
