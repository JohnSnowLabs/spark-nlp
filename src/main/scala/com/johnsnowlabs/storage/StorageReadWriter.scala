package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.util.LruMap
import org.rocksdb.WriteBatch

trait StorageReadWriter[A] extends StorageWriter[A] {

  this: StorageReader[A] =>

  @transient private val toBeWritten: LruMap[String, A] = new LruMap[String, A](writeBufferSize)

  def add(word: String, content: A): Unit = {
    if (toBeWritten.getSize >= writeBufferSize) {
      flush(new WriteBatch())
    }
    toBeWritten.update(word, Some(content))
  }

  override def lookup(index: String): Option[A] = {
    toBeWritten.get(index).orElse(_lookup(index))
  }

  override def flush(batch: WriteBatch): Unit = {
    toBeWritten.foreach{case (word, content) =>
      put(batch, word, content)
    }
    super.flush(batch)
    toBeWritten.clear()
    if (connection.isConnected)
      connection.reconnectReadWrite
  }

  override def close(): Unit = {
    flush(new WriteBatch())
    super.close()
  }

}
