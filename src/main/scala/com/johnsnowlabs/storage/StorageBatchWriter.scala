package com.johnsnowlabs.storage

import org.rocksdb.WriteBatch

trait StorageBatchWriter[A] extends StorageWriter[A] {

  protected val autoFlushAfter: Int

  private var localBatch = new WriteBatch()

  override def flush(batch: WriteBatch): Unit = {
    super.flush(batch)
    localBatch = new WriteBatch()
  }

  def add(word: String, content: A): Unit = {
    /** calling .trim because we always trim in reader */
    put(localBatch, word, content)
    if (getUpdatesCount >= autoFlushAfter)
      flush(localBatch)
  }

  override def close(): Unit = {
    if (getUpdatesCount > 0)
      flush(localBatch)

    super.close()
  }

}
