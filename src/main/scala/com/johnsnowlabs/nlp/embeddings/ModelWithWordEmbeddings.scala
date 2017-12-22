package com.johnsnowlabs.nlp.embeddings

import java.io.File
import java.nio.file.{CopyOption, Files, Paths, StandardCopyOption}
import java.util.concurrent.Executor

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

  def setDims(nDims: Int) = set(this.nDims, nDims)
  def setIndexPath(path: String) = set(this.indexPath, path)

  lazy val embeddings: Option[WordEmbeddings] = get(indexPath).map { path =>
    // Have to copy file because RockDB changes it and Spark rises Exception
    val src = SparkFiles.get(path)
    val workPath = src + "_work"
    FileUtil.deepCopy(new File(src), new File(workPath), null, true)

    WordEmbeddings(workPath, $(nDims))
  }

  override def close(): Unit = {
    if (embeddings.nonEmpty)
      embeddings.get.close()
  }

  def deserializeEmbeddings(path: String, spark: SparkContext): Unit = {
    val src = getEmbeddingsSerializedPath(path).toString

    if (new java.io.File(src).exists()) {
      WordEmbeddingsClusterHelper.copyIndexToCluster(src, spark)
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
