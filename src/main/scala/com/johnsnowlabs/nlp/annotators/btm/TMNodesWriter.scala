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

package com.johnsnowlabs.nlp.annotators.btm

import java.io.{ByteArrayOutputStream, ObjectOutputStream}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageBatchWriter}

class TMNodesWriter(
                    override protected val connection: RocksDBConnection
                  ) extends StorageBatchWriter[TrieNode] {

  def toBytes(content: TrieNode): Array[Byte] = {
    val stream: ByteArrayOutputStream = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(stream)
    oos.writeObject(content)
    oos.close()
    stream.toByteArray
  }

  def add(word: Int, value: TrieNode): Unit = {
    super.add(word.toString, value)
  }

  override protected def writeBufferSize: Int = 10000
}
