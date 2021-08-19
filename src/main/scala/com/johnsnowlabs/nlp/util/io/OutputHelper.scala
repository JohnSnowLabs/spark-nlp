/*
 * Copyright 2017-2021 John Snow Labs
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
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkFiles

import java.io.{File, FileWriter, PrintWriter}
import java.nio.charset.StandardCharsets
import scala.language.existentials


object OutputHelper {

  private lazy val fileSystem = ConfigHelper.getFileSystem

  private def logsFolder: String = ConfigLoader.getConfigStringValue(ConfigHelper.annotatorLogFolder)

  private lazy val isDBFS = fileSystem.getScheme.equals("dbfs")

  private var targetPath: Path = _

  var historyLog: Array[String] = Array()

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
      if (logsFolder.startsWith("s3")) SparkFiles.getRootDirectory() + "/tmp/logs" else logsFolder
    } else {
      outputLogsPath
    }
  }

  def exportLogFileToS3(): Unit = {
    try {
      if (isDBFS) {
        val charset = StandardCharsets.ISO_8859_1
        val outputStream = fileSystem.create(targetPath)
        historyLog.foreach(log => outputStream.write(log.getBytes(charset)))
        outputStream.close()
        historyLog = Array()
      }
      if (logsFolder.startsWith("s3")) {
        val awsGateway = new AWSGateway(ConfigLoader.getConfigStringValue(ConfigHelper.logAccessKeyId),
          ConfigLoader.getConfigStringValue(ConfigHelper.logSecretAccessKey),
          ConfigLoader.getConfigStringValue(ConfigHelper.logSessionToken),
          ConfigLoader.getConfigStringValue(ConfigHelper.logAwsProfileName),
          ConfigLoader.getConfigStringValue(ConfigHelper.logAwsRegion), "proprietary")

          val bucket = ConfigLoader.getConfigStringValue(ConfigHelper.logS3BucketKey)
          val sourceFilePath = targetPath.toString
          val s3FilePath = ConfigLoader.getConfigStringValue(ConfigHelper.annotatorLogFolder).substring("s3://".length) +
            "/" + sourceFilePath.split("/").last

          awsGateway.copyInputStreamToS3(bucket, s3FilePath, sourceFilePath)
      }
    } catch {
      case e: Exception => println(s"Warning couldn't export log on DBFS or S3 because of error: ${e.getMessage}")
    }
  }

}
