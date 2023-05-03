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

package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.client.CloudResources
import com.johnsnowlabs.client.util.CloudHelper
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkFiles

import java.io.{File, FileWriter, PrintWriter}
import java.nio.charset.StandardCharsets
import scala.util.{Failure, Success, Try}

object OutputHelper {

  private lazy val fileSystem = getFileSystem

  private lazy val sparkSession = ResourceHelper.spark

  def getFileSystem: FileSystem = {
    FileSystem.get(sparkSession.sparkContext.hadoopConfiguration)
  }
  def getFileSystem(resource: String): FileSystem = {
    val resourcePath = new Path(parsePath(resource))
    FileSystem.get(resourcePath.toUri, sparkSession.sparkContext.hadoopConfiguration)
  }

  def parsePath(path: String): String = {
    val pathPrefix = path.split("://").head
    pathPrefix match {
      case "s3" => path.replace("s3", "s3a")
      case "file" => {
        val pattern = """^file:(/+)""".r
        pattern.replaceAllIn(path, "file:///")
      }
      case _ => path
    }
  }

  def doesPathExists(resource: String): (Boolean, Option[Path]) = {
    val fileSystem = OutputHelper.getFileSystem(resource)
    var modifiedPath = resource

    fileSystem.getScheme match {
      case "file" =>
        val path = new Path(resource)
        var exists = Try {
          fileSystem.exists(path)
        } match {
          case Success(value) => value
          case Failure(_) => false
        }

        if (!exists) {
          modifiedPath = resource.replaceFirst("//+", "///")
          exists = Try {
            fileSystem.exists(new Path(modifiedPath))
          } match {
            case Success(value) => value
            case Failure(_) => false
          }
        }

        if (!exists) {
          modifiedPath = resource.replaceFirst("/+", "//")
          exists = Try {
            fileSystem.exists(new Path(modifiedPath))
          } match {
            case Success(value) => value
            case Failure(_) => false
          }
        }

        if (!exists) {
          val pattern = """^file:/*""".r
          modifiedPath = pattern.replaceAllIn(resource, "")
          exists = Try {
            fileSystem.exists(new Path(modifiedPath))
          } match {
            case Success(value) => value
            case Failure(_) => false
          }
        }

        if (exists) {
          (exists, Some(new Path(modifiedPath)))
        } else (exists, None)
      case _ => {
        val exists = Try {
          val modifiedPath = parsePath(resource)
          fileSystem.exists(new Path(modifiedPath))
        } match {
          case Success(value) => value
          case Failure(_) => false
        }

        if (exists) {
          (exists, Some(new Path(modifiedPath)))
        } else (exists, None)
      }
    }

  }

  private def getLogsFolder: String =
    ConfigLoader.getConfigStringValue(ConfigHelper.annotatorLogFolder)

  private lazy val isDBFS = fileSystem.getScheme.equals("dbfs")

  private var targetPath: Path = _

  private var historyLog: Array[String] = Array()

  def writeAppend(uuid: String, content: String, outputLogsPath: String): Unit = {
    val targetFolder = getTargetFolder(outputLogsPath)
    targetPath = new Path(targetFolder, uuid + ".log")
    if (isDBFS) {
      historyLog = historyLog ++ Array(content)
    } else {
      if (CloudHelper.isCloudPath(targetFolder)) {
        fileSystem.createNewFile(targetPath)
        val fo = fileSystem.append(targetPath)
        val writer = new PrintWriter(fo, true)
        writer.append(content + System.lineSeparator())
        writer.close()
        fo.close()
      } else {
        if (!fileSystem.exists(new Path(targetFolder))) fileSystem.mkdirs(new Path(targetFolder))
        val fo = new File(targetPath.toUri.getRawPath)
        val writer = new FileWriter(fo, true)
        writer.append(content + System.lineSeparator())
        writer.close()
      }
    }
  }

  private def getTargetFolder(outputLogsPath: String): String = {
    val path = if (outputLogsPath.isEmpty) getLogsFolder else outputLogsPath
    if (CloudHelper.isCloudPath(path)) SparkFiles.getRootDirectory() + "/tmp/logs" else path
  }

  def exportLogFile(outputLogsPath: String): Unit = {
    try {
      if (isDBFS) {
        val charset = StandardCharsets.ISO_8859_1
        val outputStream = fileSystem.create(targetPath)
        historyLog
          .map(log => log + System.lineSeparator())
          .foreach(log => outputStream.write(log.getBytes(charset)))
        outputStream.close()
        historyLog = Array()
      }

      CloudResources.storeLogFileInCloudStorage(outputLogsPath, targetPath.toString)
    } catch {
      case e: Exception =>
        println(
          s"Warning couldn't export log on DBFS or Cloud Storage because of error: ${e.getMessage}")
    }
  }

}
