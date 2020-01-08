package com.johnsnowlabs.storage

import org.rocksdb.WriteBatch

trait StorageBatchWriter[A] extends StorageWriter[A] {

  private var localBatch = new WriteBatch()
  private var batchSize = 0

  def add(word: String, content: A): Unit = {
    /** calling .trim because we always trim in reader */
    put(localBatch, word, content)
    batchSize += 1
    if (batchSize >= writeBufferSize)
      flush(localBatch)
  }

  override def flush(batch: WriteBatch): Unit = {
    super.flush(batch)
    localBatch = new WriteBatch()
    batchSize = 0
  }

  override def close(): Unit = {
    if (batchSize > 0)
      flush(localBatch)

    super.close()
  }

}
