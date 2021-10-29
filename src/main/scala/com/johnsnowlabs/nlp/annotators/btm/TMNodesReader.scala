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

import java.io.{ByteArrayInputStream, ObjectInputStream}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader}

class TMNodesReader(
                     override val connection: RocksDBConnection,
                     override protected val caseSensitiveIndex: Boolean
                  ) extends StorageReader[TrieNode] {

  override def emptyValue: TrieNode = TrieNode(0, isLeaf = true, 0, 0)

  def lookup(index: Int): TrieNode = {
    super.lookup(index.toString).get
  }

  override def fromBytes(bytes: Array[Byte]): TrieNode = {
    val ois = new ObjectInputStream(new ByteArrayInputStream(bytes))
    val value = ois.readObject.asInstanceOf[TrieNode]
    ois.close()
    value
  }

  override protected def readCacheSize: Int = 50000

}
