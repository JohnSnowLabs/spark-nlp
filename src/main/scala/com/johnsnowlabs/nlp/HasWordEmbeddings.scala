package com.johnsnowlabs.nlp

import java.io.File
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.embeddings.{SparkWordEmbeddings, WordEmbeddings, WordEmbeddingsFormat}
import org.apache.hadoop.fs.{FileSystem, Path}
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

  @transient
  private var sparkEmbeddings: SparkWordEmbeddings = null

  def embeddings: Option[WordEmbeddings] = get(indexPath).map { path =>
    if (sparkEmbeddings == null)
      sparkEmbeddings = new SparkWordEmbeddings(path, $(nDims))

    sparkEmbeddings.wordEmbeddings
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

    if (fs.exists(src)) {
      val embeddings = SparkWordEmbeddings(spark, src.toUri.toString, 0, WordEmbeddingsFormat.SPARKNLP)
      setIndexPath(embeddings.clusterFilePath.toString)
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

  override def onWrite(path: String, spark: SparkSession): Unit = {
    serializeEmbeddings(path, spark.sparkContext)
  }
}
