/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

  def merge(word: String, content: A): Unit = {
    merge(localBatch, word, content)
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
