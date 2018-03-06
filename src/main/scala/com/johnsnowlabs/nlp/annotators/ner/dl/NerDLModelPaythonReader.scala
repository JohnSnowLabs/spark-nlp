package com.johnsnowlabs.nlp.annotators.ner.dl

import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.ml.tensorflow.DatasetEncoderParams
import com.johnsnowlabs.nlp.embeddings.{SparkWordEmbeddings, WordEmbeddingsFormat}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.tensorflow.SavedModelBundle

/*
  Reads Tensorflow Model stored in Python
 */
object NerDLModelPythonReader {
  val tagsFile = "tags.csv"
  val charsFile = "chars.csv"
  val embeddingsMetaFile = "embeddings.meta"
  val embeddingsFile = "embeddings"

  private def readTags(folder: String, spark: SparkSession): List[String] = {
    spark.sparkContext.textFile(Paths.get(folder, tagsFile).toString).collect.toList
  }

  private def readChars(folder: String, spark: SparkSession): List[Char] = {
    val lines = spark.sparkContext.textFile(Paths.get(folder, charsFile).toString)
      .collect

    lines(0).toCharArray.toList
  }

  private def readEmbeddingsHead(folder: String, spark: SparkSession): Int = {
    val metaFile = Paths.get(folder, embeddingsFile).toString
    spark.sparkContext.textFile(metaFile).collect.apply(0).toInt
  }

  private def readEmbeddings(folder: String, spark: SparkSession, embeddingsDim: Int): SparkWordEmbeddings = {

    SparkWordEmbeddings(
      spark.sparkContext,
      Paths.get(folder, embeddingsFile).toString,
      embeddingsDim,
      WordEmbeddingsFormat.BINARY)
  }

  def readBundle(folder: String, spark: SparkSession): SavedModelBundle = {
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_bundle")
      .toAbsolutePath.toString

    fs.copyToLocalFile(new Path(folder), new Path(tmpFolder))

    val bundle = SavedModelBundle.load(tmpFolder)
    Files.delete(Paths.get(tmpFolder))

    bundle
  }


  def read(folder: String, spark: SparkSession): NerDLModel = {
    val labels = readTags(folder, spark)
    val chars = readChars(folder, spark)
    val embeddingsDim = readEmbeddingsHead(folder, spark)
    val embeddings = readEmbeddings(folder, spark, embeddingsDim)

    val settings = DatasetEncoderParams(labels, chars)
    val bundle = readBundle(folder, spark)

    new NerDLModel()
      .setDims(embeddingsDim)
      .setIndexPath(embeddings.clusterFilePath.toString)
      .setParams(settings)
      .setSession(bundle.session, bundle.graph)
  }
}
