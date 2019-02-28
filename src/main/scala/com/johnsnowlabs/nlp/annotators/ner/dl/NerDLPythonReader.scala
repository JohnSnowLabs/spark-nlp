package com.johnsnowlabs.nlp.annotators.ner.dl

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.ml.tensorflow.{DatasetEncoderParams, NerDatasetEncoder, TensorflowNer, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.embeddings.{ClusterWordEmbeddings, WordEmbeddingsFormat}
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

  private def readEmbeddingsHead(folder: String): Int = {
    val metaFile = Paths.get(folder, embeddingsMetaFile).toString
    Source.fromFile(metaFile).getLines().toList.head.toInt
  }

  private def readEmbeddings(
                              folder: String,
                              spark: SparkSession,
                              embeddingsDim: Int,
                              normalize: Boolean,
                              format: WordEmbeddingsFormat.Format
                            ): ClusterWordEmbeddings = {
    ClusterWordEmbeddings(
      spark.sparkContext,
      Paths.get(folder, embeddingsFile).toString,
      embeddingsDim,
      normalize,
      format,
      "python_tf_model"
    )
  }

  def readLocal(folder: String,
                dim: Int,
                useBundle: Boolean = false,
                verbose: Verbose.Level = Verbose.All,
                tags: Array[String] = Array.empty[String]): TensorflowNer = {

    val labels = readTags(folder)
    val chars = readChars(folder)
    val settings = DatasetEncoderParams(labels, chars,
      Array.fill(dim)(0f).toList, dim)
    val encoder = new NerDatasetEncoder(settings)
    val tf = TensorflowWrapper.read(folder, zipped=false, useBundle, tags)

    new TensorflowNer(tf, encoder, 32, verbose)
  }

  def read(
            folder: String,
            dim: Int,
            spark: SparkSession,
            useBundle: Boolean = false,
            tags: Array[String] = Array.empty[String]): NerDLModel = {

    val uri = new java.net.URI(folder.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_bundle")
      .toAbsolutePath.toString

    fs.copyToLocalFile(new Path(folder), new Path(tmpFolder))

    val nerModel = readLocal(tmpFolder, dim, useBundle, tags = tags)
    FileHelper.delete(tmpFolder)

    new NerDLModel()
      .setTensorflow(nerModel.tensorflow)
      .setDatasetParams(nerModel.encoder.params)
  }
}
