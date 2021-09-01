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

package com.johnsnowlabs.nlp.embeddings

import java.nio.{ByteBuffer, ByteOrder}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageBatchWriter}

class WordEmbeddingsWriter(
                                override val connection: RocksDBConnection,
                                caseSensitiveIndex: Boolean,
                                dimension: Int,
                                maxCacheSize: Int,
                                writeBuffer: Int
                          )
  extends StorageBatchWriter[Array[Float]] with ReadsFromBytes {

  override protected def writeBufferSize: Int = writeBuffer

  override def toBytes(content: Array[Float]): Array[Byte] = {
    val buffer = ByteBuffer.allocate(content.length * 4)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    for (value <- content) {
      buffer.putFloat(value)
    }
    buffer.array()
  }

}