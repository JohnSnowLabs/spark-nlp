package com.johnsnowlabs.nlp.embeddings

import java.io._
import java.nio.ByteBuffer
import org.slf4j.LoggerFactory
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

  /**
    * Indexes Word embeddings in CSV Format
    */
  def indexText(source: Iterator[String], dbFile: String): Unit = {
    TextIndexer.index(source, dbFile)
  }

  /**
    * Indexes Word embeddings in CSV Text File
    */
  def indexText(source: String, dbFile: String): Unit ={
    TextIndexer.index(source, dbFile)
  }


  def indexBinary(source: DataInputStream, dbFile: String): Unit = {
    BinaryIndexer.index(source, dbFile)
  }

  /**
    * Indexes Binary formatted file
    */
  def indexBinary(source: String, dbFile: String): Unit = {
    BinaryIndexer.index(source, dbFile)
  }
}




private[embeddings] object TextIndexer {

  def index(source: Iterator[String], dbFile: String): Unit = {
    val indexer = RocksDbIndexer(dbFile, Some(1000))

    try {
      for (line <- source) {
        val items = line.split(" ")
        val word = items(0)
        val embeddings = items.drop(1).map(i => i.toFloat)
        indexer.add(word, embeddings)
      }
    } finally {
      indexer.close()
    }
  }

  def index(source: String, dbFile: String): Unit = {
    val lines = Source.fromFile(source).getLines()
    index(lines, dbFile)
  }
}


private[embeddings] object BinaryIndexer {

  private val logger = LoggerFactory.getLogger("WordEmbeddings")

  def index(source: DataInputStream, dbFile: String): Unit = {
    val indexer = RocksDbIndexer(dbFile, Some(1000))

    try {
      // File Header
      val numWords = Integer.parseInt(readString(source))
      val vecSize = Integer.parseInt(readString(source))

      // File Body
      for (i <- 0 until numWords) {
        val word = readString(source)

        // Unit Vector
        val vector = readFloatVector(source, vecSize)
        indexer.add(word, vector)
      }

      logger.info(s"Loaded $numWords words, vector size $vecSize")
    } finally {
      indexer.close()
    }
  }

  def index(source: String, dbFile: String): Unit = {

    val ds = new DataInputStream(new BufferedInputStream(new FileInputStream(source), 1 << 15))

    try {
      index(ds, dbFile)
    } finally {
      ds.close()
    }
  }

  /**
    * Read a string from the binary model (System default should be UTF-8):
    */
  private def readString(ds: DataInputStream): String = {
    val byteBuffer = new ByteArrayOutputStream()

    var isEnd = false
    while (!isEnd) {
      val byteValue = ds.readByte()
      if ((byteValue != 32) && (byteValue != 10)) {
        byteBuffer.write(byteValue)
      } else if (byteBuffer.size() > 0) {
        isEnd = true
      }
    }

    val word = byteBuffer.toString()
    byteBuffer.close()
    word
  }

  /**
    * Read a Vector - Array of Floats from the binary model:
    */
  private def readFloatVector(ds: DataInputStream, vectorSize: Int): Array[Float] = {
    // Read Bytes
    val vectorBuffer = Array.fill[Byte](4 * vectorSize)(0)
    ds.read(vectorBuffer)

    // Convert Bytes to Floats
    WordEmbeddingsIndexer.fromBytes(vectorBuffer)
  }
}