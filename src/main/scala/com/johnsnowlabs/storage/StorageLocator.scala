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

package com.johnsnowlabs.storage

import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

case class StorageLocator(database: String, storageRef: String, sparkSession: SparkSession) {

  private lazy val fileSystem = FileSystem.get(sparkSession.sparkContext.hadoopConfiguration)

  private val clusterTmpLocation: String = {
    val tmpLocation = getTmpLocation
    if (tmpLocation.matches("s3[a]?:/.*")) {
      tmpLocation
    } else {
      fileSystem.getScheme match {
        case "abfss" =>
          if (tmpLocation.startsWith("abfss:")) {
            tmpLocation
          } else {
            "file:///" + tmpLocation
          }
        case _ =>
          val tmpLocationPath = new Path(tmpLocation)
          fileSystem.mkdirs(tmpLocationPath)
          fileSystem.deleteOnExit(tmpLocationPath)
          tmpLocation
      }
    }
  }

  val clusterFileName: String = { StorageHelper.resolveStorageName(database, storageRef) }

  val clusterFilePath: Path = {
    if (!getTmpLocation.matches("s3[a]?:/.*")) {
      val scheme = Option(new Path(clusterTmpLocation).toUri.getScheme).getOrElse("")
      scheme match {
        case "dbfs" | "hdfs" => mergePaths()
        case "file" =>
          val uri = fileSystem.getUri.toString
          if (uri.startsWith("abfss:")) { mergePaths() }
          else { mergePaths(withFileSystem = true) }
        case "abfss" => mergePaths()
        case _ => mergePaths(withFileSystem = true)
      }
    } else {
      new Path(clusterTmpLocation + "/" + clusterFileName)
    }
  }

  private def mergePaths(withFileSystem: Boolean = false): Path = {
    if (withFileSystem) {
      Path.mergePaths(
        new Path(fileSystem.getUri.toString + clusterTmpLocation),
        new Path("/" + clusterFileName))
    } else Path.mergePaths(new Path(clusterTmpLocation), new Path("/" + clusterFileName))
  }

  val destinationScheme: String = fileSystem.getScheme

  private def getTmpLocation: String = {
    ConfigLoader.getConfigStringValue(ConfigHelper.storageTmpDir)
  }

}

object StorageLocator {
  def getStorageSerializedPath(path: String, folder: String, withinStorage: Boolean): Path =
    Path.mergePaths(new Path(path), new Path((if (withinStorage) "/storage/" else "/") + folder))
}
