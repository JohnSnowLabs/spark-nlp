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
package com.johnsnowlabs.partition

import com.johnsnowlabs.reader.SparkNLPReader
import org.apache.spark.sql.DataFrame

import java.net.URL
import scala.collection.JavaConverters._

class Partition(params: java.util.Map[String, String] = new java.util.HashMap()) {

  private val sparkNLPReader = new SparkNLPReader(params)

  def partition(path: String): DataFrame = {
    if (isUrl(path)) {
      return sparkNLPReader.html(path)
    }

    val contentTypeOpt = Option(params.get("content_type"))

    val reader = contentTypeOpt match {
      case Some(contentType) => getReaderByContentType(contentType)
      case None => getReaderByExtension(path)
    }

    reader(path)
  }

  private def getReaderByContentType(contentType: String): String => DataFrame = {
    contentType match {
      case "text/plain" => sparkNLPReader.txt
      case "text/html" => sparkNLPReader.html
      case "message/rfc822" => sparkNLPReader.email
      case "application/msword" |
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document" =>
        sparkNLPReader.doc
      case "application/vnd.ms-excel" |
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" =>
        sparkNLPReader.xls
      case "application/vnd.ms-powerpoint" |
          "application/vnd.openxmlformats-officedocument.presentationml.presentation" =>
        sparkNLPReader.ppt
      case "application/pdf" => sparkNLPReader.pdf
      case _ => throw new IllegalArgumentException(s"Unsupported content type: $contentType")
    }
  }

  /** Selects the reader based on file extension */
  private def getReaderByExtension(path: String): String => DataFrame = {
    val extension = getFileExtension(path)
    extension match {
      case "txt" => sparkNLPReader.txt
      case "html" | "htm" => sparkNLPReader.html
      case "eml" | "msg" => sparkNLPReader.email
      case "doc" | "docx" => sparkNLPReader.doc
      case "xls" | "xlsx" => sparkNLPReader.xls
      case "ppt" | "pptx" => sparkNLPReader.ppt
      case "pdf" => sparkNLPReader.pdf
      case _ => throw new IllegalArgumentException(s"Unsupported file type: $extension")
    }
  }

  def partition(urls: Array[String]): DataFrame = {
    if (urls.isEmpty) throw new IllegalArgumentException("URL array is empty")
    sparkNLPReader.html(urls)
  }

  def partition(urls: java.util.List[String]): DataFrame = {
    partition(urls.asScala.toArray)
  }

  private def getFileExtension(path: String): String = {
    path.split("\\.").lastOption.map(_.toLowerCase).getOrElse("")
  }

  private def isUrl(path: String): Boolean = {
    try {
      val url = new URL(path)
      url.getProtocol == "http" || url.getProtocol == "https"
    } catch {
      case _: Exception => false
    }
  }

}

object Partition {
  def apply(params: Map[String, String] = Map.empty): Partition = {
    new Partition(mapAsJavaMap(params))
  }
}
