package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorModel
import org.apache.hadoop.fs.{FileSystem, Path}
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

  lazy val embeddings: Option[WordEmbeddings] = {
    get(indexPath).map { path =>
      WordEmbeddings(SparkFiles.get(path), $(nDims))
    }
  }

  override def close(): Unit = {
    if (embeddings.nonEmpty)
      embeddings.get.close()
  }

  def deserializeEmbeddings(path: String, spark: SparkContext): Unit = {
    val src = getEmbeddingsSerializedPath(path).toString

    if (new java.io.File(src).exists()) {
      spark.addFile(src)
      set(indexPath, src)
    }
  }

  def serializeEmbeddings(path: String, spark: SparkContext): Unit = {
    if (isDefined(indexPath)) {
      val index = new Path(SparkFiles.get($(indexPath)))
      val fs = FileSystem.get(spark.hadoopConfiguration)

      val dst = getEmbeddingsSerializedPath(path)
      fs.copyFromLocalFile(index, dst)
    }
  }

  def getEmbeddingsSerializedPath(path: String) = Path.mergePaths(new Path(path), new Path("/embeddings"))
}
