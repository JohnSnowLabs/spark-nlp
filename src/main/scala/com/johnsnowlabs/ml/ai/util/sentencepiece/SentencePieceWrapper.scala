/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.ai.util.sentencepiece

import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

private[johnsnowlabs] class SentencePieceWrapper(var sppModel: Array[Byte]) extends Serializable {

  @transient private var mspp: SentencePieceProcessor = _

  def getSppModel: SentencePieceProcessor = {

    if (mspp == null) {
      val spp = new SentencePieceProcessor()
      spp.loadFromSerializedProto(sppModel)
      mspp = spp
    }
    mspp
  }

}

object SentencePieceWrapper {

  def read(path: String): SentencePieceWrapper = {
    val byteArray = Files.readAllBytes(Paths.get(path))
    val sppWrapper = new SentencePieceWrapper(byteArray)
    val spp = new SentencePieceProcessor()
    spp.loadFromSerializedProto(byteArray)

    sppWrapper.mspp = spp
    sppWrapper
  }
}

trait WriteSentencePieceModel {
  def writeSentencePieceModel(
      path: String,
      spark: SparkSession,
      spp: SentencePieceWrapper,
      suffix: String,
      filename: String): Unit = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp folder
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath
      .toString

    val sppFile = Paths.get(tmpFolder, filename).toString

    // 2. Save Tensorflow state
    FileUtils.writeByteArrayToFile(new File(sppFile), spp.sppModel)
    // 3. Copy to dest folder
    fs.copyFromLocalFile(new Path(sppFile), new Path(path))

    // 4. Remove tmp folder
    FileUtils.deleteDirectory(new File(tmpFolder))
  }
}

trait ReadSentencePieceModel {
  val sppFile: String

  def readSentencePieceModel(
      path: String,
      spark: SparkSession,
      suffix: String,
      filename: String): SentencePieceWrapper = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp directory
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath
      .toString

    // 2. Copy to local dir
    fs.copyToLocalFile(new Path(path, filename), new Path(tmpFolder))

    val sppModelFilePath = new Path(tmpFolder, filename)

    val byteArray = Files.readAllBytes(Paths.get(sppModelFilePath.toString))
    val sppWrapper = new SentencePieceWrapper(byteArray)
    sppWrapper
  }
}
