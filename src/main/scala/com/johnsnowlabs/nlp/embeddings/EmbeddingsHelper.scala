package com.johnsnowlabs.nlp.embeddings

import java.io.FileNotFoundException

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkFiles

object EmbeddingsHelper {


  def load(
                      path: String,
                      spark: SparkSession,
                      format: String,
                      embeddingsRef: String,
                      nDims: Int,
                      caseSensitiveEmbeddings: Boolean
                    ): ClusterWordEmbeddings = {
    import WordEmbeddingsFormat._
    load(
      path,
      spark,
      str2frm(format),
      nDims,
      caseSensitiveEmbeddings,
      Option(embeddingsRef)
    )
  }

  def load(
                    path: String,
                    spark: SparkSession,
                    format: WordEmbeddingsFormat.Format,
                    nDims: Int,
                    caseSensitiveEmbeddings: Boolean,
                    embeddingsRef: Option[String]): ClusterWordEmbeddings = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val src = new Path(path)

    if (fs.exists(src)) {
      ClusterWordEmbeddings(
        spark.sparkContext,
        src.toUri.toString,
        nDims,
        caseSensitiveEmbeddings,
        format,
        embeddingsRef
      )
    } else {
      throw new FileNotFoundException(s"embeddings not found in $path")
    }

  }

  def load(
                    indexPath: String,
                    nDims: Int,
                    caseSensitive: Boolean
                    ): ClusterWordEmbeddings = {
    new ClusterWordEmbeddings(indexPath, nDims, caseSensitive)
  }

  def getFromAnnotator(annotator: ModelWithWordEmbeddings): ClusterWordEmbeddings = {
    annotator.getClusterEmbeddings
  }

  def getClusterPath(embeddingsRef: String): String = {
    Path.mergePaths(new Path("/embd_"), new Path(embeddingsRef)).toString
  }

  protected def save(path: String, spark: SparkSession, indexPath: String): Unit = {
    val index = new Path(SparkFiles.get(indexPath))

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = new Path(path)

    save(fs, index, dst)
  }

  def save(path: String, embeddings: ClusterWordEmbeddings, spark: SparkSession): Unit = {
    EmbeddingsHelper.save(path, spark, embeddings.clusterFilePath.toString)
  }

  def save(fs: FileSystem, index: Path, dst: Path): Unit = {
    fs.copyFromLocalFile(false, true, index, dst)
  }

}
