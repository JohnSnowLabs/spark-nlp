package com.johnsnowlabs.nlp.embeddings

import java.nio.file.Files
import java.util.UUID

import com.johnsnowlabs.nlp.util.SparkNlpConfigKeys
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.sql.SparkSession


trait AnnotatorWithWordEmbeddings extends AutoCloseable { this: Estimator[_] =>
  val sourceEmbeddingsPath = new Param[String](this, "sourceEmbeddingsPath", "Word embeddings file")
  val embeddingsFormat = new IntParam(this, "embeddingsFormat", "Word vectors file format")
  val embeddingsNDims = new IntParam(this, "embeddingsNDims", "Number of dimensions for word vectors")

  val embeddingsFolder = new Param[String](this, "embeddingsFolder",
    "Folder to store Embeddings Index")

  private val defaultFolder = spark.sparkContext.getConf
    .getOption(SparkNlpConfigKeys.embeddingsFolder).getOrElse("embeddings/")

  setDefault(this.embeddingsFolder -> defaultFolder)


  def setEmbeddingsSource(path: String, nDims: Int, format: WordEmbeddingsFormat.Format) = {
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
    set(this.embeddingsNDims, nDims)
  }

  def setEmbeddingsFolder(path: String) = set(this.embeddingsFolder, path)

  def fillModelEmbeddings[T <: ModelWithWordEmbeddings](model: T): T = {
    if (!isDefined(sourceEmbeddingsPath)) {
      return model
    }

    val file = "/" + new Path(localPath).getName
    val path = Path.mergePaths(new Path($(embeddingsFolder)), new Path(file))
    hdfs.copyFromLocalFile(new Path(localPath), path)

    model.setDims($(embeddingsNDims))

    model.setIndexPath(path.toUri.toString)

    model
  }

  lazy val embeddings: Option[WordEmbeddings] = {
    get(sourceEmbeddingsPath).map(_ => WordEmbeddings(localPath, $(embeddingsNDims)))
  }

  private lazy val localPath: String = {
    val path = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_idx")
      .toAbsolutePath.toString

    if ($(embeddingsFormat) == WordEmbeddingsFormat.SparkNlp.id) {
      hdfs.copyToLocalFile(new Path($(sourceEmbeddingsPath)), new Path(path))
    } else {
      indexEmbeddings(path)
    }

    path
  }

  private lazy val spark: SparkSession = {
    SparkSession
      .builder()
      .getOrCreate()
  }

  private lazy val hdfs: FileSystem = {
    FileSystem.get(spark.sparkContext.hadoopConfiguration)
  }

  private def indexEmbeddings(localFile: String): Unit = {
    val formatId = $(embeddingsFormat)
    if (formatId == WordEmbeddingsFormat.Text.id) {
      val lines = spark.sparkContext.textFile($(sourceEmbeddingsPath)).toLocalIterator
      WordEmbeddingsIndexer.indexText(lines, localFile)
    } else if (formatId == WordEmbeddingsFormat.Binary.id) {
      val streamSource = spark.sparkContext.binaryFiles($(sourceEmbeddingsPath)).toLocalIterator.toList.head._2
      val stream = streamSource.open()
      try {
        WordEmbeddingsIndexer.indexBinary(stream, localFile)
      }
      finally {
        stream.close()
      }
    }
    else if (formatId == WordEmbeddingsFormat.SparkNlp.id) {
        hdfs.copyToLocalFile(new Path($(sourceEmbeddingsPath)), new Path(localFile))
    }
  }

  override def close(): Unit = {
    if (embeddings.nonEmpty)
      embeddings.get.close()
  }
}
