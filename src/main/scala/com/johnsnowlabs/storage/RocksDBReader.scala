package com.johnsnowlabs.storage

import java.nio.{ByteBuffer, ByteOrder}

import com.johnsnowlabs.nlp.util.LruMap
import spire.ClassTag

abstract class RocksDBReader[A: ClassTag](
                                 connection: RocksDBConnection,
                                 caseSensitiveIndex: Boolean,
                                 lruCacheSize: Int = 100000) extends AutoCloseable {

  connection.connectReadOnly

  @transient val lru = new LruMap[String, Option[Array[A]]](lruCacheSize)

  protected val emptyValue: A

  def getConnection: RocksDBConnection = connection

  protected def getFromBuffer(buffer: ByteBuffer, index: Int): A // wrapper.getFloat(...)

  def fromBytes(source: Array[Byte]): Array[A] = {
    val wrapper = ByteBuffer.wrap(source)
    wrapper.order(ByteOrder.LITTLE_ENDIAN)
    val result = Array.fill[A](source.length / 4)(emptyValue)

    for (i <- result.indices) {
      result(i) = getFromBuffer(wrapper, i * 4)
    }
    result
  }

  private def lookupByIndex(index: String): Option[Array[A]] = {
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

  def lookup(index: String): Option[Array[A]] = {
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