package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.util.LruMap
import spire.ClassTag

abstract class StorageReader[A: ClassTag](
                                 connection: RocksDBConnection,
                                 caseSensitiveIndex: Boolean,
                                 lruCacheSize: Int = 100000) extends AutoCloseable {

  connection.connectReadOnly

  @transient val lru = new LruMap[String, Option[A]](lruCacheSize)

  def emptyValue: A

  def getConnection: RocksDBConnection = connection

  def fromBytes(source: Array[Byte]): A

  private def lookupByIndex(index: String): Option[A] = {
    lazy val resultLower = connection.getDb.get(index.trim.toLowerCase.getBytes())
    lazy val resultUpper = connection.getDb.get(index.trim.toUpperCase.getBytes())
    lazy val resultExact = connection.getDb.get(index.trim.getBytes())

    if (resultExact != null)
      Some(fromBytes(resultExact))
    else if (caseSensitiveIndex && resultLower != null)
      Some(fromBytes(resultLower))
    else if (caseSensitiveIndex && resultExact != null)
      Some(fromBytes(resultExact))
    else if (caseSensitiveIndex && resultUpper != null)
      Some(fromBytes(resultUpper))
    else
      None
  }

  def lookup(index: String): Option[A] = {
    synchronized {
      lru.getOrElseUpdate(index, lookupByIndex(index))
    }
  }

  def containsIndex(index: String): Boolean = {
    val wordBytes = index.trim.getBytes()
    connection.getDb.get(wordBytes) != null ||
      (connection.getDb.get(index.trim.toLowerCase.getBytes()) != null) ||
      (connection.getDb.get(index.trim.toUpperCase.getBytes()) != null)
  }

  override def close(): Unit = {
    connection.close()
  }

}