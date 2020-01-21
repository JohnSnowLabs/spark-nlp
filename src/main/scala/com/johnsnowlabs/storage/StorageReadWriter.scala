package com.johnsnowlabs.storage

import org.rocksdb.WriteBatch
import scala.collection.mutable.{Map => MMap}

trait StorageReadWriter[A] extends StorageWriter[A] {

  this: StorageReader[A] =>

  @transient private val readableWriteBuffer: MMap[String, A] = MMap.empty[String, A]
  private var bufferCounter = 0

  def add(word: String, content: A): Unit = {
    if (bufferCounter >= writeBufferSize) {
      flush(new WriteBatch())
      bufferCounter = 0
    }
    bufferCounter += 1
    readableWriteBuffer.update(word, content)
  }

  override def lookup(index: String): Option[A] = {
    readableWriteBuffer.get(index).orElse(_lookup(index))
  }

  override def flush(batch: WriteBatch): Unit = {
    readableWriteBuffer.foreach{case (word, content) =>
      put(batch, word, content)
    }
    super.flush(batch)
    readableWriteBuffer.clear()
    if (connection.isConnected)
      connection.reconnectReadWrite
  }

  override def close(): Unit = {
    flush(new WriteBatch())
    this.clear()
    super.close()
  }

}
