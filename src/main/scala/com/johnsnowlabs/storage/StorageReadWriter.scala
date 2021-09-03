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
