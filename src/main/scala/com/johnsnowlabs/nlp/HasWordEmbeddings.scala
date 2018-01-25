package com.johnsnowlabs.nlp

import java.io.File
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsClusterHelper}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.ivy.util.FileUtil
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkContext, SparkFiles}


/**
  * Base class for models that uses Word Embeddings.
  * This implementation is based on RocksDB so it has a compact RAM usage
  *
  * Corresponding Approach have to implement AnnotatorWithWordEmbeddings
   */
trait HasWordEmbeddings extends AutoCloseable with ParamsAndFeaturesWritable {

  val nDims = new IntParam(this, "nDims", "Number of embedding dimensions")
  val indexPath = new Param[String](this, "indexPath", "File that stores Index")

  def setDims(nDims: Int): this.type = set(this.nDims, nDims)
  def setIndexPath(path: String): this.type = set(this.indexPath, path)

  lazy val embeddings: Option[WordEmbeddings] = get(indexPath).map { path =>
    // Have to copy file because RockDB changes it and Spark rises Exception
    val src = SparkFiles.get(path)
    val workPath = src + "_work"
    if (!new File(workPath).exists())
      FileUtil.deepCopy(new File(src), new File(workPath), null, false)

    WordEmbeddings(workPath, $(nDims))
  }

  override def close(): Unit = {
    if (embeddings.nonEmpty)
      embeddings.get.close()
  }

  def moveFolderFiles(folderSrc: String, folderDst: String): Unit = {
    for (file <- new File(folderSrc).list()) {
      Files.move(Paths.get(folderSrc, file), Paths.get(folderDst, file))
    }

    Files.delete(Paths.get(folderSrc))
  }

  def deserializeEmbeddings(path: String, spark: SparkContext): Unit = {
    val fs = FileSystem.get(spark.hadoopConfiguration)

    val src = getEmbeddingsSerializedPath(path)

    // 1. Copy to local file
    val localPath = WordEmbeddingsClusterHelper.createLocalPath()
    if (fs.exists(src)) {
      fs.copyToLocalFile(src, new Path(localPath))

      // 2. Move files from localPath/embeddings to localPath
      moveFolderFiles(localPath + "/embeddings", localPath)

      // 2. Copy local file to cluster
      WordEmbeddingsClusterHelper.copyIndexToCluster(localPath, spark)

      // 3. Set correct path
      val fileName = WordEmbeddingsClusterHelper.getClusterFileName(localPath).toString
      setIndexPath(fileName)
    }
  }

  def serializeEmbeddings(path: String, spark: SparkContext): Unit = {
    if (isDefined(indexPath)) {
      val index = new Path(SparkFiles.get($(indexPath)))
      val fs = FileSystem.get(spark.hadoopConfiguration)

      val dst = getEmbeddingsSerializedPath(path)
      fs.copyFromLocalFile(false, true, index, dst)
    }
  }

  def getEmbeddingsSerializedPath(path: String): Path = Path.mergePaths(new Path(path), new Path("/embeddings"))

  override def onWritten(path: String, spark: SparkSession): Unit = {
    deserializeEmbeddings(path, spark.sparkContext)
  }

}
