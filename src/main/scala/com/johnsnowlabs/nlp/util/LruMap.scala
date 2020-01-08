package com.johnsnowlabs.nlp.util

import scala.collection.mutable


class LruMap[TKey, TValue](maxCacheSize: Int) {
  private val cache = mutable.Map[TKey, TValue]()
  private val lru = mutable.PriorityQueue[KeyPriority]()(KeyPriorityOrdering)

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

  def update(key: TKey, value: => Option[TValue]): Option[TValue] = {
    if (value.isDefined) {
      val isNewKey = !cache.contains(key)
      if (isNewKey && getSize >= maxCacheSize)
        deleteOne()
      else if (isNewKey)
        size += 1

      cache(key) = value.get
      priorityCounter += 1
      lru.enqueue(KeyPriority(key, priorityCounter))
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

  def get(key: TKey): Option[TValue] = {
    cache.get(key)
  }

  case class KeyPriority(key: TKey, priority: Int)

  object KeyPriorityOrdering extends Ordering[KeyPriority] {
    override def compare(x: KeyPriority, y: KeyPriority): Int = x.priority.compareTo(y.priority)
  }
}
