/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.partition.util

import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import java.nio.charset.Charset
import java.nio.file.Files

object PartitionHelper {

  def datasetWithBinaryFile(sparkSession: SparkSession, contentPath: String): DataFrame = {
    import sparkSession.implicits._
    val binaryFilesRDD = sparkSession.sparkContext.binaryFiles(contentPath)
    val byteArrayRDD = binaryFilesRDD.map { case (path, portableDataStream) =>
      val byteArray = portableDataStream.toArray()
      (path, byteArray)
    }
    byteArrayRDD.toDF("path", "content")
  }

  def datasetWithTextFile(sparkSession: SparkSession, contentPath: String): DataFrame = {
    import sparkSession.implicits._
    val textFilesRDD = sparkSession.sparkContext.wholeTextFiles(contentPath)
    textFilesRDD
      .toDF("path", "content")
  }

  def isStringContent(contentType: String): Boolean = {
    contentType match {
      case "text/plain" | "text/html" | "text/markdown" | "application/xml" | "text/csv" |
          "url" =>
        true
      case _ => false
    }
  }

  def datasetWithTextFileEncoding(
      sparkSession: SparkSession,
      contentPath: String,
      encoding: String): DataFrame = {
    import sparkSession.implicits._
    val fs = new java.io.File(contentPath)
    val files =
      if (fs.isDirectory) fs.listFiles.filter(_.isFile).map(_.getPath)
      else Array(contentPath)
    val fileContents = files.map { path =>
      val content =
        new String(Files.readAllBytes(java.nio.file.Paths.get(path)), Charset.forName(encoding))
      (path, content)
    }
    sparkSession.sparkContext.parallelize(fileContents).toDF("path", "content")
  }

}
