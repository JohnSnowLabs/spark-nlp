package com.johnsnowlabs.nlp.embeddings

import java.io.File
import java.nio.file.Files
import java.util.UUID

import com.johnsnowlabs.nlp.util.SparkNlpConfigKeys
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.sql.SparkSession


/**
  * Trait for models that want to use Word Embeddings
  *
  * Corresponding Approach have to implement AnnotatorWithWordEmbeddings
   */
trait ModelWithWordEmbeddings extends AutoCloseable {
  this: Model[_] =>

  val nDims = new IntParam(this, "nDims", "Number of embedding dimensions")
  val indexPath = new Param[String](this, "indexPath", "File that stores Index")

  def setDims(nDims: Int) = set(this.nDims, nDims)
  def setIndexPath(path: String) = set(this.indexPath, path)

  private lazy val spark = {
    SparkSession.builder().getOrCreate()
  }

  private lazy val hdfs = {
    FileSystem.get(spark.sparkContext.hadoopConfiguration)
  }

  private lazy val embeddingsFile: String = {
    val localFile = if (!new File($(indexPath)).exists()) {
      val localPath = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "embedddings_idx")
        .toAbsolutePath.toString

      hdfs.copyToLocalFile(new Path($(indexPath)), new Path(localPath))
      localPath
    } else {
      $(indexPath)
    }

    val crcFiles = new File(localFile).listFiles().filter(f => f.getName.endsWith(".crc"))
    for (file <- crcFiles) {
      file.delete()
    }

    localFile
  }

  lazy val embeddings: Option[WordEmbeddings] = {
    get(indexPath).map { path =>
      WordEmbeddings(embeddingsFile, $(nDims))
    }
  }

  override def close(): Unit = {
    if (embeddings.nonEmpty)
      embeddings.get.close()
  }

  def deserializeEmbeddings(path: String): Unit = {
    if (isDefined(indexPath)) {
      val embeddingsFolder = spark.conf.getOption(SparkNlpConfigKeys.embeddingsFolder)
      if (embeddingsFolder.isDefined) {
        val dst = new Path(embeddingsFolder.get)
        val file = getEmbeddingsSerializedPath(path).getName

        val indexFile = new Path(dst.toString, file)
        setIndexPath(indexFile.toString)
      }

      try {
        // ToDo make files comparision
        if (!hdfs.exists(new Path($(indexPath))))
          FileUtil.copy(hdfs, getEmbeddingsSerializedPath(path), hdfs, new Path($(indexPath)), false, spark.sparkContext.hadoopConfiguration)
      }
      catch {
        case e: Exception =>
          throw new Exception(s"Set spark option ${SparkNlpConfigKeys.embeddingsFolder} to store embeddings", e)
      }
    }
  }

  def serializeEmbeddings(path: String): Unit = {
    if (isDefined(indexPath)) {
      val dst = getEmbeddingsSerializedPath(path)
      if (hdfs.exists(dst)) {
        hdfs.delete(dst, true)
      }

      hdfs.copyFromLocalFile(new Path(embeddingsFile), dst)
    }
  }

  def getEmbeddingsSerializedPath(path: String) = Path.mergePaths(new Path(path), new Path("/embeddings"))
}
