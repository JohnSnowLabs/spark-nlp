package com.johnsnowlabs.nlp.embeddings

import java.io.Closeable
import java.nio.ByteBuffer

import com.johnsnowlabs.nlp.util.LruMap
import org.rocksdb._

import scala.io.Source


object WordEmbeddingsIndexer {

  private[embeddings] def toBytes(embeddings: Array[Float]): Array[Byte] = {
    val buffer = ByteBuffer.allocate(embeddings.length * 4)
    for (value <- embeddings) {
      buffer.putFloat(value)
    }
    buffer.array()
  }

  private[embeddings] def fromBytes(source: Array[Byte]): Array[Float] = {
    val wrapper = ByteBuffer.wrap(source)
    val result = Array.fill[Float](source.length / 4)(0f)

    for (i <- 0 until result.length) {
      result(i) = wrapper.getFloat(i * 4)
    }
    result
  }

  def indexGlove(source: Iterator[String], dbFile: String): Unit = {
    val options = new Options()
    options.setCreateIfMissing(true)
    options.setWriteBufferSize(20 * 1 << 20)

    RocksDB.loadLibrary()
    val writeOptions = new WriteOptions()

    val db = RocksDB.open(options, dbFile)
    var batch = new WriteBatch()
    try {
      var batchSize = 0
      for (line <- source) {
        val items = line.split(" ")
        val word = items(0)
        val embeddings = items.drop(1).map(i => i.toFloat)
        batch.put(word.getBytes, toBytes(embeddings))

        batchSize += 1
        if (batchSize % 1000 == 0) {
          db.write(writeOptions, batch)
          batch.close()
          batch = new WriteBatch()
          batchSize == 0
        }
      }

      db.write(writeOptions, batch)
      batch.close()
    } finally {
      db.close()
    }
  }

  def indexGlove(source: String, dbFile: String): Unit = {
    val lines = Source.fromFile(source).getLines()
    indexGlove(lines, dbFile)
  }
}

case class WordEmbeddings(dbFile: String,
                          nDims: Int,
                          cacheSizeMB: Int = 100,
                          lruCacheSize: Int = 100000) extends Closeable{
  val options = new Options()
  options.setRowCache(new LRUCache(cacheSizeMB * 1 << 20))
  RocksDB.loadLibrary()

  val db = RocksDB.openReadOnly(options, dbFile)

  val zeroArray = Array.fill[Float](nDims)(0f)

  val lru = new LruMap[String, Array[Float]](lruCacheSize)

  private def getEmbeddingsFromDb(word: String): Array[Float] = {
    val result = db.get(word.toLowerCase.trim.getBytes())
    if (result == null)
      zeroArray
    else
      WordEmbeddingsIndexer.fromBytes(result)
  }

  def getEmbeddings(word: String): Array[Float] = {
    lru.getOrElseUpdate(word, getEmbeddingsFromDb(word))
  }

  override def close(): Unit = {
    db.close()
  }
}
