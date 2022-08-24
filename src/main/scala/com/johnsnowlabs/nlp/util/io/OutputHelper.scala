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

import com.johnsnowlabs.client.aws.AWSGateway
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkFiles

import java.io.{File, FileWriter, PrintWriter}
import java.nio.charset.StandardCharsets
import scala.language.existentials

object OutputHelper {

  private lazy val fileSystem = getFileSystem

  private lazy val sparkSession = ResourceHelper.spark

  def getFileSystem: FileSystem = {
    FileSystem.get(sparkSession.sparkContext.hadoopConfiguration)
  }

  def getFileSystem(resource: String): (FileSystem, Path) = {
    val resourcePath = new Path(resource)
    val fileSystem =
      FileSystem.get(resourcePath.toUri, sparkSession.sparkContext.hadoopConfiguration)

    (fileSystem, resourcePath)
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
      if (!fileSystem.exists(new Path(targetFolder))) fileSystem.mkdirs(new Path(targetFolder))

      if (fileSystem.getScheme.equals("file")) {
        val fo = new File(targetPath.toUri.getRawPath)
        val writer = new FileWriter(fo, true)
        writer.append(content + System.lineSeparator())
        writer.close()
      } else {
        fileSystem.createNewFile(targetPath)
        val fo = fileSystem.append(targetPath)
        val writer = new PrintWriter(fo, true)
        writer.append(content + System.lineSeparator())
        writer.close()
        fo.close()
      }
    }
  }

  private def getTargetFolder(outputLogsPath: String): String = {
    if (outputLogsPath.isEmpty) {
      if (getLogsFolder.startsWith("s3")) SparkFiles.getRootDirectory() + "/tmp/logs"
      else getLogsFolder
    } else {
      if (outputLogsPath.startsWith("s3")) SparkFiles.getRootDirectory() + "/tmp/logs"
      else outputLogsPath
    }
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

      if (outputLogsPath.startsWith("s3")) {
        val sourceFilePath = targetPath.toString
        val s3Bucket = outputLogsPath.replace("s3://", "").split("/").head
        val s3Path = "s3:/" + outputLogsPath.substring(s"s3://$s3Bucket".length) + "/"

        storeFileInS3(sourceFilePath, s3Bucket, s3Path)
      } else if (getLogsFolder.startsWith("s3")) {
        val sourceFilePath = targetPath.toString
        val s3Bucket = ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalS3BucketKey)
        val s3Path = ConfigLoader.getConfigStringValue(ConfigHelper.annotatorLogFolder) + "/"

        storeFileInS3(sourceFilePath, s3Bucket, s3Path)
      }

    } catch {
      case e: Exception =>
        println(s"Warning couldn't export log on DBFS or S3 because of error: ${e.getMessage}")
    }
  }

  def storeFileInS3(sourceFilePath: String, s3Bucket: String, s3Path: String): Unit = {
    val awsGateway = new AWSGateway(credentialsType = "proprietary")
    val s3FilePath = s"""${s3Path.substring("s3://".length)}${sourceFilePath.split("/").last}"""

    awsGateway.copyInputStreamToS3(s3Bucket, s3FilePath, sourceFilePath)
  }

}
