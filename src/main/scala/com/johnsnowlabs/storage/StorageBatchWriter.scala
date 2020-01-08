package com.johnsnowlabs.storage

import org.rocksdb.WriteBatch

trait StorageBatchWriter[A] extends StorageWriter[A] {

  private var localBatch = new WriteBatch()

  override def flush(batch: WriteBatch): Unit = {
    super.flush(batch)
    localBatch = new WriteBatch()
  }

  def add(word: String, content: A): Unit = {
    /** calling .trim because we always trim in reader */
    put(localBatch, word, content)
    if (getBatchSize >= writeBufferSize)
      flush(localBatch)
  }

  override def close(): Unit = {
    if (getBatchSize > 0)
      flush(localBatch)

    super.close()
  }

}
