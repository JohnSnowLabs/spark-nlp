package com.johnsnowlabs.nlp.annotators.ner.dl

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.ml.tensorflow.{DatasetEncoder, DatasetEncoderParams, TensorflowWrapper}
import com.johnsnowlabs.nlp.embeddings.{SparkWordEmbeddings, WordEmbeddingsFormat}
import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.tensorflow.SavedModelBundle

import scala.io.Source


object NerDLModelPythonReader {
  val embeddingsMetaFile = "embeddings.meta"
  val embeddingsFile = "embeddings"
  val tagsFile = "tags.csv"
  val charsFile = "chars.csv"


  private def readTags(folder: String): List[String] = {
    Source.fromFile(Paths.get(folder, tagsFile).toString).getLines().toList
  }

  private def readChars(folder: String): List[Char] = {
    val lines = Source.fromFile(Paths.get(folder, charsFile).toString).getLines()
    lines.toList.head.toCharArray.toList
  }

  private def readBundle(folder: String): SavedModelBundle = {
    SavedModelBundle.load(Paths.get(folder).toString)
  }

  private def readEmbeddingsHead(folder: String, spark: SparkSession): Int = {
    val metaFile = Paths.get(folder, embeddingsMetaFile).toString
    Source.fromFile(metaFile).toList.head.toInt
  }

  private def readEmbeddings(folder: String, spark: SparkSession, embeddingsDim: Int): SparkWordEmbeddings = {
    SparkWordEmbeddings(
      spark.sparkContext,
      Paths.get(folder, embeddingsFile).toString,
      embeddingsDim,
      WordEmbeddingsFormat.BINARY)
  }

  def read(folder: String, spark: SparkSession): NerDLModel = {
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_bundle")
      .toAbsolutePath.toString

    fs.copyToLocalFile(new Path(folder), new Path(tmpFolder))

    val embeddingsDim = readEmbeddingsHead(folder, spark)
    val embeddings = readEmbeddings(folder, spark, embeddingsDim)
    val labels = readTags(folder)
    val chars = readChars(folder)
    val settings = DatasetEncoderParams(labels, chars)
    val encoder = new DatasetEncoder(embeddings.wordEmbeddings.getEmbeddings, settings)
    val bundle = readBundle(folder)
    val tf = new TensorflowWrapper(bundle.session, bundle.graph)

    FileUtils.deleteDirectory(new File(tmpFolder))

    new NerDLModel()
      .setTensorflow(tf)
      .setDatasetParams(encoder.params)
  }
}
