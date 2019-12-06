package com.johnsnowlabs.storage

import java.io.File

import com.johnsnowlabs.nlp.util.LruMap
import org.apache.spark.SparkFiles
import org.rocksdb.RocksDB

/*
  1. Copy Embeddings to local tmp file
  2. Index Embeddings if need
  3. Copy Index to cluster
  4. Open RocksDb based Embeddings on local index (lazy)
 */
abstract class RocksDBReader[A](val fileName: String, val caseSensitive: Boolean, lruCacheSize: Int = 100000)
  extends RocksDBConnection with Serializable with AutoCloseable {

  @transient val lru = new LruMap[String, Option[Array[A]]](lruCacheSize)

  lazy protected val localPath: String = StorageHelper.getLocalPath(fileName)

  def findLocalDb: RocksDB = {
    if (isOpen)
      getDb
    else if (new File(localPath).exists()) {
      openReadOnly(localPath)
    }
    else {
      val localFromClusterPath = SparkFiles.get(fileName)
      require(new File(localFromClusterPath).exists(), s"Storage not found under given ref: $fileName\n" +
        s" This usually means:\n1. You have not loaded any storage under such ref\n2." +
        s" You are trying to use cluster mode without a proper shared filesystem.\n3. source was not provided to Storage creation" +
        s"\n4. If you are trying to utilize Storage defined elsewhere, make sure you it's appropriate ref. ")
      openReadOnly(localFromClusterPath)
    }
  }

  protected def customizedLookup(index: String): Option[Array[A]]

  def lookupIndex(index: String): Option[Array[A]] = {
    synchronized {
      lru.getOrElseUpdate(index, customizedLookup(index))
    }
  }

  def containsIndex(index: String): Boolean = {
    val wordBytes = index.trim.getBytes()
    findLocalDb.get(wordBytes) != null ||
      (findLocalDb.get(index.trim.toLowerCase.getBytes()) != null) ||
      (findLocalDb.get(index.trim.toUpperCase.getBytes()) != null)

  }

  override def close(): Unit = {
    super.close()
  }

}