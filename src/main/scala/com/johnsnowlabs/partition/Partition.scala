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

import com.johnsnowlabs.reader.{HTMLElement, SparkNLPReader}
import org.apache.spark.sql.DataFrame

import java.net.URL
import scala.collection.JavaConverters._
import scala.util.Try

class Partition(params: java.util.Map[String, String] = new java.util.HashMap())
    extends Serializable {

  private var outputColumn = "partition"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Result column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  def partition(
      path: String,
      headers: java.util.Map[String, String] = new java.util.HashMap()): DataFrame = {
    val sparkNLPReader = new SparkNLPReader(params, headers)
    sparkNLPReader.setOutputColumn(outputColumn)
    if (isUrl(path) && getContentType.isDefined) {
      return sparkNLPReader.html(path)
    }

    val reader = getContentType match {
      case Some(contentType) => getReaderByContentType(contentType, sparkNLPReader)
      case None => getReaderByExtension(path, sparkNLPReader)
    }

    reader(path)
  }

  def partitionStringContent(
      input: String,
      headers: java.util.Map[String, String] = new java.util.HashMap()): Seq[HTMLElement] = {
    require(getContentType.isDefined, "ContentType cannot be empty.")
    val sparkNLPReader = new SparkNLPReader(params, headers)
    sparkNLPReader.setOutputColumn(outputColumn)
    val reader = getReaderForStringContent(getContentType.get, sparkNLPReader)
    reader(input)
  }

  def partitionBytesContent(input: Array[Byte]): Seq[HTMLElement] = {
    require(getContentType.isDefined, "ContentType cannot be empty.")
    val sparkNLPReader = new SparkNLPReader(params)
    sparkNLPReader.setOutputColumn(outputColumn)
    val reader = getReaderForBytesContent(getContentType.get, sparkNLPReader)
    reader(input)
  }

  private def getReaderByContentType(
      contentType: String,
      sparkNLPReader: SparkNLPReader): String => DataFrame = {
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

  private def getReaderForStringContent(
      contentType: String,
      sparkNLPReader: SparkNLPReader): String => Seq[HTMLElement] = {
    contentType match {
      case "text/plain" => sparkNLPReader.txtToHTMLElement
      case "text/html" => sparkNLPReader.htmlToHTMLElement
      case "url" => sparkNLPReader.urlToHTMLElement
      case _ => throw new IllegalArgumentException(s"Unsupported content type: $contentType")
    }
  }

  private def getReaderForBytesContent(
      contentType: String,
      sparkNLPReader: SparkNLPReader): Array[Byte] => Seq[HTMLElement] = {
    contentType match {
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
      case _ => throw new IllegalArgumentException(s"Unsupported content type: $contentType")
//      case "application/pdf" => sparkNLPReader.pdf
    }

  }

  private def getReaderByExtension(
      path: String,
      sparkNLPReader: SparkNLPReader): String => DataFrame = {
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

  def partitionUrls(urls: Array[String], headers: Map[String, String] = Map.empty): DataFrame = {
    if (urls.isEmpty) throw new IllegalArgumentException("URL array is empty")
    val sparkNLPReader = new SparkNLPReader(params, headers.asJava)
    sparkNLPReader.html(urls)
  }

  def partitionUrlsJava(
      urls: java.util.List[String],
      headers: java.util.Map[String, String] = new java.util.HashMap()): DataFrame = {
    partitionUrls(urls.asScala.toArray, headers.asScala.toMap)
  }

  def partitionText(text: String): DataFrame = {
    val sparkNLPReader = new SparkNLPReader(params)
    sparkNLPReader.txtContent(text)
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

  private def getContentType: Option[String] = {
    Seq("content_type", "contentType")
      .flatMap(key => Option(params.get(key)))
      .flatMap(value => Try(value).toOption)
      .headOption
  }

}

object Partition {
  def apply(params: Map[String, String] = Map.empty): Partition = {
    new Partition(mapAsJavaMap(params))
  }
}
