package com.johnsnowlabs.nlp.annotators.common

import java.io.{Closeable, File}
import java.nio.ByteBuffer

import org.fusesource.leveldbjni.JniDBFactory.{bytes, factory}
import org.iq80.leveldb.Options

import scala.io.Source


object WordEmbeddingsIndexer {

  private[common] def toBytes(embeddings: Array[Float]): Array[Byte] = {
    val buffer = ByteBuffer.allocate(embeddings.length * 4)
    for (value <- embeddings) {
      buffer.putFloat(value)
    }
    buffer.array()
  }

  private[common] def fromBytes(source: Array[Byte]): Array[Float] = {
    val wrapper = ByteBuffer.wrap(source)
    val result = Array.fill[Float](source.length / 4)(0f)

    for (i <- 0 until result.length) {
      result(i) = wrapper.getFloat(i * 4)
    }
    result
  }
  
  def indexGloveToLevelDb(source: String, dbFile: String): Unit = {
    val options = new Options()
    options.createIfMissing(true)
    val db = factory.open(new File(dbFile), options)
    var batch = db.createWriteBatch
    try {
      var batchSize = 0
      for (line <- Source.fromFile(source).getLines()) {
        val items = line.split(" ")
        val word = items(0)
        val embeddings = items.drop(1).map(i => i.toFloat)
        batch.put(bytes(word), toBytes(embeddings))

        batchSize += 1
        if (batchSize % 1000 == 0) {
          db.write(batch)
          batch.close()
          batch = db.createWriteBatch()
          batchSize == 0
        }
      }

      db.write(batch)
      batch.close()
    } finally {
      db.close()
    }
  }
}

case class WordEmbeddings(levelDbFile: String,
                          nDims: Int,
                          cacheSizeMB: Int = 100,
                          lruCacheSize: Int = 100000) extends Closeable{
  val options = new Options()
  options.cacheSize(cacheSizeMB * 1048576) // 100 Mb
  options.createIfMissing(true)
  val db = factory.open(new File(levelDbFile), options)
  val zeroArray = Array.fill[Float](nDims)(0f)

  val lru = new LruMap[String, Array[Float]](lruCacheSize)

  private def getEmbeddingsFromDb(word: String): Array[Float] = {
    val result = db.get(bytes(word.toLowerCase.trim))
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
