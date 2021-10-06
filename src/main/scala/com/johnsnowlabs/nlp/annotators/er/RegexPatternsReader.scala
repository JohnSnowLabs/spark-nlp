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
package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader}

import java.io.{ByteArrayInputStream, ObjectInputStream}

class RegexPatternsReader(protected val connection: RocksDBConnection) extends StorageReader[Seq[String]] {

  protected val caseSensitiveIndex: Boolean = false

  protected def readCacheSize: Int = 50000

  def emptyValue: Seq[String] = Seq()

  def fromBytes(source: Array[Byte]): Seq[String] = {
    val objectInputStream = new ObjectInputStream(new ByteArrayInputStream(source))
    val value = objectInputStream.readObject().asInstanceOf[Seq[String]]
    objectInputStream.close()
    value
  }


}
