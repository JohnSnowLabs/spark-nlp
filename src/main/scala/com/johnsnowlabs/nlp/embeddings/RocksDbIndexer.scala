package com.johnsnowlabs.nlp.embeddings

import java.io.Closeable

import org.rocksdb._


private [embeddings] case class RocksDbIndexer(dbFile: String, autoFlashAfter: Option[Integer] = None) extends Closeable{
  val options = new Options()
  options.setCreateIfMissing(true)
  options.setCompressionType(CompressionType.LZ4_COMPRESSION)
  options.setWriteBufferSize(20 * 1 << 20)

  RocksDB.loadLibrary()
  val writeOptions = new WriteOptions()

  val db = RocksDB.open(options, dbFile)
  var batch = new WriteBatch()
  var batchSize = 0

  def flush() = {
    db.write(writeOptions, batch)
    batch.close()
    batch = new WriteBatch()
    batchSize = 0
  }

  def add(word: String, vector: Array[Float]) = {
    batch.put(word.getBytes, WordEmbeddingsIndexer.toBytes(vector))
    batchSize += 1

    if (autoFlashAfter.isDefined) {
      if (batchSize >= autoFlashAfter.get)
        flush()
    }
  }

  override def close(): Unit = {
    if (batchSize > 0)
      flush()

    db.close
  }
}
