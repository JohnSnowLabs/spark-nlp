package com.johnsnowlabs.nlp.embeddings

import java.io._
import java.nio.ByteBuffer

import com.johnsnowlabs.storage.RocksDBReadWrite
import org.slf4j.LoggerFactory

import scala.io.Source


class WordEmbeddingsReadWrite(dbFile: String, mode: String, autoFlashAfter: Option[Integer] = None)
  extends RocksDBReadWrite[Float](dbFile, mode, autoFlashAfter) {

  override protected val emptyValue: Float = 0f

  override protected def addToBuffer(buffer: ByteBuffer, content: Float): Unit = {
    buffer.putFloat(content)
  }

  override protected def getFromBuffer(buffer: ByteBuffer, index: Int): Float = {
    buffer.getFloat(index)
  }

}


object WordEmbeddingsTextIndexer {

  def index(source: Iterator[String], dbFile: String, autoFlashAfter: Option[Integer] = Some(1000)): Unit = {
    val indexer = new WordEmbeddingsReadWrite(dbFile, "w", autoFlashAfter)

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
    val sourceFile = Source.fromFile(source)("UTF-8")
    val lines = sourceFile.getLines()
    index(lines, dbFile)
    sourceFile.close()
  }
}


object WordEmbeddingsBinaryIndexer {

  private val logger = LoggerFactory.getLogger("WordEmbeddings")

  def index(source: DataInputStream, dbFile: String, autoFlashAfter: Option[Integer] = Some(1000)): Unit = {
    val indexer = new WordEmbeddingsReadWrite(dbFile, "w", autoFlashAfter)

    try {
      // File Header
      val numWords = Integer.parseInt(readString(source))
      val vecSize = Integer.parseInt(readString(source))

      // File Body
      for (i <- 0 until numWords) {
        val word = readString(source)

        // Unit Vector
        val vector = readFloatVector(source, vecSize, indexer)
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
  private def readFloatVector(ds: DataInputStream, vectorSize: Int, indexer: WordEmbeddingsReadWrite): Array[Float] = {
    // Read Bytes
    val vectorBuffer = Array.fill[Byte](4 * vectorSize)(0)
    ds.read(vectorBuffer)

    // Convert Bytes to Floats
    indexer.fromBytes(vectorBuffer)
  }
}