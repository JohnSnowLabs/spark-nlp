/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.ner.dl

import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.ml.tensorflow.{DatasetEncoderParams, NerDatasetEncoder, TensorflowNer, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.storage.{RocksDBConnection, StorageHelper}
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
                              normalize: Boolean
                            ): RocksDBConnection = {
    StorageHelper.load(
      Paths.get(folder, embeddingsFile).toString,
      spark,
      "python_tf_model",
      "python_tf_ref",
      false
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
    val (tf, _) = TensorflowWrapper.read(folder, zipped = false, useBundle, tags)

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
      .setModelIfNotSet(spark, nerModel.tensorflow)
      .setDatasetParams(nerModel.encoder.params)
  }
}
