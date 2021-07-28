package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.client.AWSGateway
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

  def processLogFile(): Unit = {
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
          ConfigLoader.getConfigStringValue(ConfigHelper.logAwsProfileName),
          ConfigLoader.getConfigStringValue(ConfigHelper.logAwsRegion), "proprietary")
        if (awsGateway.credentials.isEmpty) {
          println("Warning couldn't export log on S3 because some credential is missing")
        } else {
          val bucket = ConfigLoader.getConfigStringValue(ConfigHelper.logS3BucketKey)
          val sourceFilePath = targetPath.toString
          val s3FilePath = ConfigLoader.getConfigStringValue(ConfigHelper.annotatorLogFolder).substring("s3://".length) +
            "/" + sourceFilePath.split("/").last
          awsGateway.copyInputStreamToS3(bucket, s3FilePath, sourceFilePath)
        }
      }
    } catch {
      case e: Exception => println(s"Warning couldn't export log on DBFS or S3 because of error: ${e.getMessage}")
    }
  }

}
