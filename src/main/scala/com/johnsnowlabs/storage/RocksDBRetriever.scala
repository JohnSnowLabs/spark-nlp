package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.util.LruMap
import org.rocksdb._

import scala.reflect.ClassTag


abstract class RocksDBRetriever[A: ClassTag](dbFile: String,
                                caseSensitive: Boolean,
                                lruCacheSize: Int = 100000) extends AutoCloseable {

  @transient private var prefetchedDB: RocksDB = _

  protected def customizedLookup(index: String): Option[Array[A]]

  protected def db: RocksDB = {
    if (Option(prefetchedDB).isDefined)
      prefetchedDB
    else {
      RocksDB.loadLibrary()
      prefetchedDB = RocksDB.openReadOnly(dbFile)
      prefetchedDB
    }
  }

  val lru = new LruMap[String, Option[Array[A]]](lruCacheSize)

  def lookupIndex(index: String): Option[Array[A]] = {
    synchronized {
      lru.getOrElseUpdate(index, customizedLookup(index))
    }
  }

  def containsIndex(index: String): Boolean = {
    val wordBytes = index.trim.getBytes()
    db.get(wordBytes) != null ||
      (db.get(index.trim.toLowerCase.getBytes()) != null) ||
      (db.get(index.trim.toUpperCase.getBytes()) != null)

  }

  override def close(): Unit = {
    if (Option(prefetchedDB).isDefined) {
      db.close()
      prefetchedDB = null
    }
  }
}
