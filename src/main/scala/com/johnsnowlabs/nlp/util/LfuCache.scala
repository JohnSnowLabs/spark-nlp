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
class LfuCache[TKey, TValue](maxSize: Int) {

  private val entries = mutable.HashMap.empty[TKey, CachedItem]
  private val frequencyList = new FrequencyList(0)

  def getSize: Int = entries.size

  def getOrElseUpdate(key: TKey, value: => TValue): TValue = {
    if (entries.contains(key)) get(key).get
    else {
      val content = value
      append(key, content)
      content
    }
  }

  def get(itemId: TKey): Option[TValue] = {
    entries.get(itemId).map { cachedItem => {
      cachedItem.bump()
      cachedItem.value
    }}
  }

  def removeLast(): Option[TValue] = {
    if (entries.isEmpty) return None

    val last = frequencyList.next.removeLast()
    entries.remove(last.key)
    Some(last.value)
  }

  private def append(key: TKey, item: => TValue): Unit = {
    if (getSize == maxSize) removeLast()

    val cachedItem = new CachedItem(key, item)
    frequencyList.add(cachedItem)
    cachedItem.bump()
    entries.put(key, cachedItem)
  }

  class FrequencyList(val frequency: Int) extends DoubleLinked[FrequencyList] {

    var previous: FrequencyList = this
    var next: FrequencyList = this

    var items: CachedItem = _

    def bump(item: CachedItem): Unit = {
      val bumpedFrequency = frequency + 1
      val linked =
        if (next.frequency == bumpedFrequency) next
        else link(new FrequencyList(bumpedFrequency))

      remove(item)
      linked.add(item)
    }

    def link(link: FrequencyList): FrequencyList = {
      link.next = next
      link.previous = this
      next.previous = link
      next = link
      link
    }

    def unlink(link: FrequencyList): FrequencyList = {
      link.previous.next = link.next
      link.next.previous = link.previous
      link
    }

    def add(item: CachedItem): Unit = {
      item.list = this
      if (items == null) item.reset()
      else items.addBefore(item)
      items = item
    }

    def remove(item: CachedItem): Unit = {
      if (frequency == 0) items = null
      else if (item.isSingle) unlink(this)
      else item.remove()

      if (items == item) items = item.next
    }

    def removeLast(): CachedItem = {
      if (items.isSingle) unlink(this).items
      else items.last.remove()
    }

  }

  class CachedItem(val key: TKey, val value: TValue) extends DoubleLinked[CachedItem] {

    var list: FrequencyList = _

    var previous: CachedItem = this
    var next: CachedItem = this

    def isSingle: Boolean =
      next == this

    def addBefore(item: CachedItem): Unit = {
      item.previous = previous
      item.next = this
      previous.next = item
      previous = item
    }

    def remove(): CachedItem = {
      previous.next = next
      next.previous = previous
      this
    }

    def reset(): Unit = {
      previous = this
      next = this
    }

    def bump(): Unit = {
      list.bump(this)
    }

    def last: CachedItem =
      previous

  }

  trait DoubleLinked[Type <: DoubleLinked[Type]] { self: Type =>

    def next: Type
    def previous: Type

    def iterate(f: Type => Unit): Unit = {
      var tail = this
      do {
        f(tail)
        tail = tail.next
      } while (tail != this)
    }

  }

}