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

package com.johnsnowlabs.nlp.util

import scala.collection.mutable

@specialized
class LruMap[TKey, TValue](maxCacheSize: Int) {
  private val cache = mutable.Map.empty[TKey, TValue]
  private val lru = mutable.PriorityQueue.empty[KeyPriority](KeyPriorityOrdering)

  private var priorityCounter = 0
  private var size = 0

  private def deleteOne(): Unit = {
    val oldest = lru.dequeue().key
    cache.remove(oldest)
  }

  def clear(): Unit = {
    cache.clear()
    lru.clear()
    priorityCounter = 0
    size = 0
  }

  def getSize: Int = {
    size
  }

  def foreach: (((TKey, TValue)) => Any) => Unit = cache.foreach

  def update(key: TKey, value: => TValue): TValue = synchronized {
    val isNewKey = !cache.contains(key)
    if (isNewKey && getSize >= maxCacheSize)
      deleteOne()
    else if (isNewKey)
      size += 1

    val content = value
    cache(key) = content
    priorityCounter += 1
    lru.enqueue(KeyPriority(key, priorityCounter))
    content
  }

  def getOrElseUpdate(key: TKey, valueCreator: => TValue): TValue = {
    val oldValue = cache.get(key)
    if (oldValue.isDefined) {
      oldValue.get
    } else {
      update(key, valueCreator)
    }
  }

  def get(key: TKey): Option[TValue] = {
    cache.get(key)
  }

  case class KeyPriority(key: TKey, priority: Int)

  object KeyPriorityOrdering extends Ordering[KeyPriority] {
    override def compare(x: KeyPriority, y: KeyPriority): Int = y.priority.compareTo(x.priority)
  }
}
