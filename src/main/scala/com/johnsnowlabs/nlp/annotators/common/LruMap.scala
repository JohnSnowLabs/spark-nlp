package com.johnsnowlabs.nlp.annotators.common

import scala.collection.mutable


class LruMap[TKey, TValue](maxCacheSize: Int) {
  val cache = mutable.Map[TKey, TValue]()
  val lru = mutable.PriorityQueue[KeyPriority]()(KeyPriorityOrdering)

  var counter = 0

  private def deleteOne(): Unit = {
    val oldest = lru.dequeue().key
    cache.remove(oldest)
  }

  def getOrElseUpdate(key: TKey, valueCreator: => TValue): TValue = {
    val oldValue = cache.get(key)
    if (oldValue.isDefined) {
      oldValue.get
    } else {
      if (cache.size >= maxCacheSize)
        deleteOne()

      val value = valueCreator
      cache(key) = value
      counter += 1
      lru.enqueue(KeyPriority(key, counter))
      value
    }
  }

  case class KeyPriority(key: TKey, priority: Int)

  object KeyPriorityOrdering extends Ordering[KeyPriority] {
    override def compare(x: KeyPriority, y: KeyPriority): Int = x.priority.compareTo(y.priority)
  }
}