package com.johnsnowlabs.nlp.embeddings

import java.io.{BufferedInputStream, ByteArrayOutputStream, DataInputStream, FileInputStream}
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.storage.{RocksDBConnection, StorageIndexer}
import com.johnsnowlabs.util.FileHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.ivy.util.FileUtil
import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory

import scala.io.Source

object WordEmbeddingsIndexer extends StorageIndexer {
  override val filesPrefix: String = "embd_"

  override val StorageFormats: EmbeddingsFormat.type = EmbeddingsFormat

  override def indexStorage(
                             storageSourcePath: String,
                             localFile: String,
                             format: StorageFormats.Value,
                             spark: SparkContext): Unit = {

    val uri = new java.net.URI(storageSourcePath.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.hadoopConfiguration)

    lazy val connection = new RocksDBConnection(localFile)

    if (format == EmbeddingsFormat.TEXT) {

      val tmpFile = Files.createTempFile(filesPrefix, ".txt").toAbsolutePath.toString
      fs.copyToLocalFile(new Path(storageSourcePath), new Path(tmpFile))
      WordEmbeddingsTextIndexer.index(tmpFile, connection)
      FileHelper.delete(tmpFile)
    }
    else if (format == EmbeddingsFormat.BINARY) {

      val tmpFile = Files.createTempFile(filesPrefix, ".bin").toAbsolutePath.toString
      fs.copyToLocalFile(new Path(storageSourcePath), new Path(tmpFile))
      WordEmbeddingsBinaryIndexer.index(tmpFile, connection)
      FileHelper.delete(tmpFile)
    }
    else if (format == EmbeddingsFormat.SPARKNLP) {

      fs.copyToLocalFile(new Path(storageSourcePath), new Path(localFile))
      val fileName = new Path(storageSourcePath).getName

      /** If we remove this deepCopy line, word storage will fail (needs research) - moving it instead of copy also fails*/
      FileUtil.deepCopy(Paths.get(localFile, fileName).toFile, Paths.get(localFile).toFile, null, true)
      FileHelper.delete(Paths.get(localFile, fileName).toString)
    }
  }
}

object WordEmbeddingsTextIndexer {

  def index(
             source: Iterator[String],
             connection: RocksDBConnection,
             autoFlushAfter: Option[Integer]
           ): Unit = {
    val indexer = new WordEmbeddingsWriter(connection,autoFlushAfter)

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

  def index(
             source: String,
             connection: RocksDBConnection,
             autoFlashAfter: Option[Integer] = Some(1000)
           ): Unit = {
    val sourceFile = Source.fromFile(source)("UTF-8")
    val lines = sourceFile.getLines()
    index(lines, connection, autoFlashAfter)
    sourceFile.close()
  }
}


object WordEmbeddingsBinaryIndexer {

  private val logger = LoggerFactory.getLogger("WordEmbeddings")

  def index(
             source: DataInputStream,
             connection: RocksDBConnection,
             autoFlashAfter: Option[Integer],
             lruCacheSize: Int): Unit = {
    val indexer = new WordEmbeddingsWriter(connection, autoFlashAfter)
    val reader = new WordEmbeddingsReader(connection, caseSensitiveIndex=true, lruCacheSize)

    try {
      // File Header
      val numWords = Integer.parseInt(readString(source))
      val vecSize = Integer.parseInt(readString(source))

      // File Body
      for (i <- 0 until numWords) {
        val word = readString(source)

        // Unit Vector
        val vector = readFloatVector(source, vecSize, reader)
        indexer.add(word, vector)
      }

      logger.info(s"Loaded $numWords words, vector size $vecSize")
    } finally {
      indexer.close()
    }
  }

  def index(
             source: String,
             connection: RocksDBConnection,
             autoFlashAfter: Option[Integer] = Some(1000),
             lruCacheSize: Int = 100000): Unit = {

    val ds = new DataInputStream(new BufferedInputStream(new FileInputStream(source), 1 << 15))

    try {
      index(ds, connection, autoFlashAfter, lruCacheSize)
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
  private def readFloatVector(ds: DataInputStream, vectorSize: Int, indexer: WordEmbeddingsReader): Array[Float] = {
    // Read Bytes
    val vectorBuffer = Array.fill[Byte](4 * vectorSize)(0)
    ds.read(vectorBuffer)

    // Convert Bytes to Floats
    indexer.fromBytes(vectorBuffer)
  }
}
