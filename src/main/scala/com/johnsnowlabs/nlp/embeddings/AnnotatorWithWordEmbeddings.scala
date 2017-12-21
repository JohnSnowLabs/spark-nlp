package com.johnsnowlabs.nlp.embeddings

import java.io.File
import java.nio.file.Files
import java.util.UUID

import com.johnsnowlabs.nlp.AnnotatorApproach
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.sql.SparkSession


/**
  * Base class for annotators that uses Word Embeddings.
  * This implementation is based on RocksDB so it has a compact RAM usage
  *
  * 1. User configures Word Embeddings by method 'setWordEmbeddingsSource'.
  * 2. During training Word Embeddings are indexed as RockDB index file.
  * 3. Than this index file is spread across the cluster.
  * 4. Every model 'ModelWithWordEmbeddings' uses local RocksDB as Word Embeddings lookup.
 */
abstract class AnnotatorWithWordEmbeddings[M <: ModelWithWordEmbeddings[M]]
  extends AnnotatorApproach[M] with AutoCloseable {

  val sourceEmbeddingsPath = new Param[String](this, "sourceEmbeddingsPath", "Word embeddings file")
  val embeddingsFormat = new IntParam(this, "embeddingsFormat", "Word vectors file format")
  val embeddingsNDims = new IntParam(this, "embeddingsNDims", "Number of dimensions for word vectors")


  def setEmbeddingsSource(path: String, nDims: Int, format: WordEmbeddingsFormat.Format) = {
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
    set(this.embeddingsNDims, nDims)
  }

  override def beforeTraining(spark: SparkSession): Unit = {
    if (isDefined(sourceEmbeddingsPath)) {
      indexEmbeddings(localPath, spark.sparkContext)
      spark.sparkContext.addFile(localPath, true)
    }
  }

  override def onTrained(model: M, spark: SparkSession): Unit = {
    if (isDefined(sourceEmbeddingsPath)) {
      model.setDims($(embeddingsNDims))

      val fileName = new File(localPath).getName
      model.setIndexPath(fileName)
    }
  }

  lazy val embeddings: Option[WordEmbeddings] = {
    get(sourceEmbeddingsPath).map(_ => WordEmbeddings(localPath, $(embeddingsNDims)))
  }

  private lazy val localPath: String = {
    Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_idx")
      .toAbsolutePath.toString
  }

  private def indexEmbeddings(localFile: String, spark: SparkContext): Unit = {
    val formatId = $(embeddingsFormat)

    if (formatId == WordEmbeddingsFormat.Text.id) {
      val lines = spark.textFile($(sourceEmbeddingsPath)).toLocalIterator
      WordEmbeddingsIndexer.indexText(lines, localFile)
    } else if (formatId == WordEmbeddingsFormat.Binary.id) {
      val streamSource = spark.binaryFiles($(sourceEmbeddingsPath)).toLocalIterator.toList.head._2
      val stream = streamSource.open()
      try {
        WordEmbeddingsIndexer.indexBinary(stream, localFile)
      }
      finally {
        stream.close()
      }
    }
    else if (formatId == WordEmbeddingsFormat.SparkNlp.id) {
      val hdfs = FileSystem.get(spark.hadoopConfiguration)
      hdfs.copyToLocalFile(new Path($(sourceEmbeddingsPath)), new Path(localFile))
    }
  }

  override def close(): Unit = {
    if (embeddings.nonEmpty)
      embeddings.get.close()
  }
}
