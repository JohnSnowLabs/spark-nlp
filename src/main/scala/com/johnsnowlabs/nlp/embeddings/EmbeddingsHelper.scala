package com.johnsnowlabs.nlp.embeddings

import java.nio.file.{Files, Paths}

import com.johnsnowlabs.storage.{RocksDBRetriever, StorageConnection, StorageHelper}
import com.johnsnowlabs.util.FileHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.ivy.util.FileUtil
import org.apache.spark.SparkContext

object EmbeddingsHelper extends StorageHelper[Float, WordEmbeddingsStorageConnection] {

  override val filesPrefix: String = "embd_"

  override val StorageFormats: EmbeddingsFormat.type = EmbeddingsFormat

  override protected def createConnection(filename: String, caseSensitive: Boolean): WordEmbeddingsStorageConnection = {
    new WordEmbeddingsStorageConnection(filename, caseSensitive)//.asInstanceOf[StorageConnection[Float, RocksDBRetriever[Float]]]
  }

  override protected def indexStorage(storageSourcePath: String, localFile: String, format: StorageFormats.Value, spark: SparkContext): Unit = {

    val uri = new java.net.URI(storageSourcePath.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.hadoopConfiguration)

    if (format == EmbeddingsFormat.TEXT) {

      val tmpFile = Files.createTempFile(filesPrefix, ".txt").toAbsolutePath.toString
      fs.copyToLocalFile(new Path(storageSourcePath), new Path(tmpFile))
      WordEmbeddingsTextIndexer.index(tmpFile, localFile)
      FileHelper.delete(tmpFile)
    }
    else if (format == EmbeddingsFormat.BINARY) {

      val tmpFile = Files.createTempFile(filesPrefix, ".bin").toAbsolutePath.toString
      fs.copyToLocalFile(new Path(storageSourcePath), new Path(tmpFile))
      WordEmbeddingsBinaryIndexer.index(tmpFile, localFile)
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
