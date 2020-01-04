package com.johnsnowlabs.storage

import org.rocksdb.WriteBatch

trait StorageReadWriter[A] extends StorageWriter[A] {

  this: StorageReader[A] =>

  def add(word: String, content: A): Unit = {
    if (getUpdatesCount >= cacheSize) {
      flush(new WriteBatch())
    }
    lru.update(word, Some(content))
    updatesCount += 1
  }

  override def flush(batch: WriteBatch): Unit = {
    lru.foreach{case (word, content) => {
      put(batch, word, content)
    }}
    super.flush(batch)
  }

  override def close(): Unit = {
    flush(new WriteBatch())
    super.close()
  }

}
