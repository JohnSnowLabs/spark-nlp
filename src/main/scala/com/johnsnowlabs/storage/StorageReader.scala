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