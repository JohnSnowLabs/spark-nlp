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

package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.pretrained.ResourceType.ResourceType
import com.johnsnowlabs.nlp.pretrained.{ResourceMetadata, ResourceType}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.util.MLWriter
import org.apache.spark.sql.Dataset

import java.io.File
import java.nio.file.Paths
import java.sql.Timestamp
import java.util.Date
import scala.util.Try

object TrainingHelper {

  def saveModel(
      name: String,
      language: Option[String],
      libVersion: Option[Version],
      sparkVersion: Option[Version],
      modelWriter: MLWriter,
      folder: String,
      category: Option[ResourceType] = Some(ResourceType.NOT_DEFINED)): Unit = {

    // 1. Get current timestamp
    val timestamp = new Timestamp(new Date().getTime)

    // 2. Create resource metadata
    val meta = new ResourceMetadata(
      name,
      language,
      libVersion,
      sparkVersion,
      true,
      timestamp,
      true,
      category = Some(category.get.toString))

    // 3. Save model to file
    val file = Paths.get(folder, meta.key).toString.replaceAllLiterally("\\", "/")
    modelWriter.save(file)

    // 4. Zip file
    val zipFile = Paths.get(folder, meta.fileName).toString
    ZipArchiveUtil.zip(file, zipFile)

    // 5. Remove original file
    FileUtils.forceDeleteOnExit(new File(file))

    // 6. Remove original file
    try {
      FileUtils.deleteDirectory(new File(file))
    } catch {
      case _: java.io.IOException => // file lock may prevent deletion, ignore and continue
    }

    // 7. Add to metadata.json info about resource
    val metadataFile = Paths.get(folder, "metadata.json").toString
    ResourceMetadata.addMetadataToFile(metadataFile, meta)
  }

  def hasColumn(dataSet: Dataset[_], column: String): Boolean = Try(dataSet(column)).isSuccess

}
