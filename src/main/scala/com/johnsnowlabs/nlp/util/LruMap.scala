package com.johnsnowlabs.nlp.util

import scala.collection.mutable


class LruMap[TKey, TValue](maxCacheSize: Int) {
  private val cache = mutable.Map[TKey, TValue]()
  private val lru = mutable.PriorityQueue[KeyPriority]()(KeyPriorityOrdering)

  private var counter = 0

  private def deleteOne(): Unit = {
    val oldest = lru.dequeue().key
    cache.remove(oldest)
  }

  def foreach: (((TKey, TValue)) => Any) => Unit = cache.foreach

  def update(key: TKey, value: => Option[TValue]): Option[TValue] = {
    if (cache.size >= maxCacheSize)
      deleteOne()

    if (value.isDefined) {
      cache(key) = value.get
      counter += 1
      lru.enqueue(KeyPriority(key, counter))
    }
    value
  }

  def getOrElseUpdate(key: TKey, valueCreator: => Option[TValue]): Option[TValue] = {
    val oldValue = cache.get(key)
    if (oldValue.isDefined) {
      oldValue
    } else {
      update(key, valueCreator)
    }
  }

  case class KeyPriority(key: TKey, priority: Int)

  object KeyPriorityOrdering extends Ordering[KeyPriority] {
    override def compare(x: KeyPriority, y: KeyPriority): Int = x.priority.compareTo(y.priority)
  }
}
