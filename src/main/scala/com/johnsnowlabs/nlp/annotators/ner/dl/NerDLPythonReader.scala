package com.johnsnowlabs.nlp.annotators.ner.dl

import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.ml.tensorflow.{DatasetEncoderParams, NerDatasetEncoder, TensorflowWrapper}
import com.johnsnowlabs.nlp.embeddings.{SparkWordEmbeddings, WordEmbeddingsFormat}
import com.johnsnowlabs.util.FileHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

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

  private def readEmbeddingsHead(folder: String, spark: SparkSession): Int = {
    val metaFile = Paths.get(folder, embeddingsMetaFile).toString
    Source.fromFile(metaFile).getLines().toList.head.toInt
  }

  private def readEmbeddings(
                              folder: String,
                              spark: SparkSession,
                              embeddingsDim: Int,
                              normalize: Boolean,
                              format: WordEmbeddingsFormat.Format
                            ): SparkWordEmbeddings = {
    SparkWordEmbeddings(
      spark.sparkContext,
      Paths.get(folder, embeddingsFile).toString,
      embeddingsDim,
      normalize,
      format)
  }

  def read(
            folder: String,
            spark: SparkSession,
            normalize: Boolean,
            format: WordEmbeddingsFormat.Format = WordEmbeddingsFormat.BINARY,
            useBundle: Boolean = false,
            tags: Array[String] = Array.empty[String]): NerDLModel = {

    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_bundle")
      .toAbsolutePath.toString

    fs.copyToLocalFile(new Path(folder), new Path(tmpFolder))

    val embeddingsDim = readEmbeddingsHead(folder, spark)
    val embeddings = readEmbeddings(folder, spark, embeddingsDim, normalize, format)
    val labels = readTags(folder)
    val chars = readChars(folder)
    val settings = DatasetEncoderParams(labels, chars)
    val encoder = new NerDatasetEncoder(embeddings.wordEmbeddings.getEmbeddings, settings)
    val tf = TensorflowWrapper.read(folder, zipped=false, useBundle, tags)

    FileHelper.delete(tmpFolder)

    new NerDLModel()
      .setTensorflow(tf)
      .setDatasetParams(encoder.params)
      .setDims(embeddingsDim)
      .setIndexPath(embeddings.clusterFilePath.toString)
      .setNormalizeEmbeddings(normalize)
  }
}
