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

import com.johnsnowlabs.nlp.util.LruMap

trait StorageReader[A] extends HasConnection {

  protected val caseSensitiveIndex: Boolean
  protected def readCacheSize: Int

  @transient private val lru = new LruMap[String, Option[A]](readCacheSize)

  def emptyValue: A

  def fromBytes(source: Array[Byte]): A

  protected def lookupDisk(index: String): Option[A] = {
    lazy val resultLower = connection.getDb.get(index.trim.toLowerCase.getBytes())
    lazy val resultUpper = connection.getDb.get(index.trim.toUpperCase.getBytes())
    lazy val resultExact = connection.getDb.get(index.trim.getBytes())

    if (resultExact != null)
      Some(fromBytes(resultExact))
    else if (!caseSensitiveIndex && resultLower != null)
      Some(fromBytes(resultLower))
    else if (!caseSensitiveIndex && resultUpper != null)
      Some(fromBytes(resultUpper))
    else
      None
  }

  protected def _lookup(index: String): Option[A] = {
    lru.getOrElseUpdate(index, lookupDisk(index))
  }

  def lookup(index: String): Option[A] = {
    _lookup(index)
  }

  def containsIndex(index: String): Boolean = {
    lookupDisk(index).isDefined
  }

  def clear(): Unit = {
    lru.clear()
  }

}