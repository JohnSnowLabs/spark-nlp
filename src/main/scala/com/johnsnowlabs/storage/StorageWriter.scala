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

import org.rocksdb.{WriteBatch, WriteOptions}

trait StorageWriter[A] extends HasConnection {

  protected def writeBufferSize: Int

  def toBytes(content: A): Array[Byte]

  def add(word: String, content: A): Unit

  protected def put(batch: WriteBatch, word: String, content: A): Unit = {
    batch.put(word.trim.getBytes, toBytes(content))
  }

  protected def merge(batch: WriteBatch, word: String, content: A): Unit = {
    batch.merge(word.trim.getBytes, toBytes(content))
  }

  def flush(batch: WriteBatch): Unit = {
    val writeOptions = new WriteOptions()
    /** Might have disconnected already */
    if (connection.isConnected) {
      connection.getDb.write(writeOptions, batch)
    }
    batch.close()
  }

  def close(): Unit

}
