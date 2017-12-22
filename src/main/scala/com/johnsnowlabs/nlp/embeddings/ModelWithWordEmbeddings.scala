package com.johnsnowlabs.nlp.embeddings

import java.io.File
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.AnnotatorModel
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.ivy.util.FileUtil
import org.apache.spark.{SparkContext, SparkFiles}
import org.apache.spark.ml.param.{IntParam, Param}


/**
  * Base class for models that uses Word Embeddings.
  * This implementation is based on RocksDB so it has a compact RAM usage
  *
  * Corresponding Approach have to implement AnnotatorWithWordEmbeddings
   */
abstract class ModelWithWordEmbeddings[M <: ModelWithWordEmbeddings[M]]
  extends AnnotatorModel[M] with AutoCloseable {

  val nDims = new IntParam(this, "nDims", "Number of embedding dimensions")
  val indexPath = new Param[String](this, "indexPath", "File that stores Index")

  def setDims(nDims: Int) = set(this.nDims, nDims).asInstanceOf[M]
  def setIndexPath(path: String) = set(this.indexPath, path).asInstanceOf[M]

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
    val localPath = WordEmbeddingsClusterHelper.createLocalPath
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

  def getEmbeddingsSerializedPath(path: String) = Path.mergePaths(new Path(path), new Path("/embeddings"))
}
