/*
 * Copyright 2017-2024 John Snow Labs
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
package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.partition.util.PartitionHelper.datasetWithBinaryFile
import jakarta.mail._
import jakarta.mail.internet.MimeMessage
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.io.{ByteArrayInputStream, InputStream}
import java.util.Properties
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class EmailReader(addAttachmentContent: Boolean = false, storeContent: Boolean = false)
    extends Serializable {

  private lazy val spark = ResourceHelper.spark
  private var outputColumn = "email"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  def email(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val emailDf = datasetWithBinaryFile(spark, filePath)
        .withColumn(outputColumn, parseEmailUDF(col("content")))
      if (storeContent) emailDf.select("path", outputColumn, "content")
      else emailDf.select("path", outputColumn)
    } else throw new IllegalArgumentException(s"Invalid filePath: $filePath")
  }

  private val parseEmailUDF = udf((data: Array[Byte]) => {
    val inputStream = new ByteArrayInputStream(data)
    parseEmailFile(inputStream)
  })

  def emailToHTMLElement(content: Array[Byte]): Seq[HTMLElement] = {
    val inputStream = new ByteArrayInputStream(content)
    parseEmailFile(inputStream)
  }

  private def parseEmailFile(inputStream: InputStream): Array[HTMLElement] = {
    val session = getJavaMailSession
    val elements = ArrayBuffer[HTMLElement]()
    val mimeMessage = new MimeMessage(session, inputStream)

    val subject = mimeMessage.getSubject
    val recipientsMetadata = retrieveRecipients(mimeMessage)
    elements += HTMLElement(ElementType.TITLE, content = subject, metadata = recipientsMetadata)

    // Recursive function to process each part based on its type
    def extractContentFromPart(part: Part): Unit = {
      val partType = classifyMimeType(part)
      partType match {
        case MimeType.TEXT_PLAIN =>
          if (part.getFileName != null && part.getFileName.nonEmpty) {
            elements += HTMLElement(
              ElementType.ATTACHMENT,
              content = part.getFileName,
              metadata = recipientsMetadata ++ Map("contentType" -> part.getContentType))
            if (addAttachmentContent) {
              elements += HTMLElement(
                ElementType.NARRATIVE_TEXT,
                content = part.getContent.toString,
                metadata = recipientsMetadata ++ Map("mimeType" -> "text/plain"))
            }
          } else {
            elements += HTMLElement(
              ElementType.NARRATIVE_TEXT,
              content = part.getContent.toString,
              metadata = recipientsMetadata ++ Map("mimeType" -> "text/plain"))
          }

        case MimeType.TEXT_HTML =>
          elements += HTMLElement(
            ElementType.NARRATIVE_TEXT,
            content = part.getContent.toString,
            metadata = recipientsMetadata ++ Map("mimeType" -> "text/html"))

        case MimeType.MULTIPART =>
          // Recursively process nested Multipart
          val nestedMultipart = part.getContent.asInstanceOf[Multipart]
          for (i <- 0 until nestedMultipart.getCount) {
            extractContentFromPart(nestedMultipart.getBodyPart(i))
          }
        case MimeType.IMAGE | MimeType.APPLICATION =>
          elements += HTMLElement(
            ElementType.ATTACHMENT,
            content = part.getFileName,
            metadata = recipientsMetadata ++ Map("contentType" -> part.getContentType))
        case MimeType.UNKNOWN =>
          // Handle any other unknown part types as uncategorized
          elements += HTMLElement(
            ElementType.UNCATEGORIZED_TEXT,
            content = "Unknown content",
            metadata = recipientsMetadata)
      }
    }

    mimeMessage.getContent match {
      case multipart: Multipart =>
        for (i <- 0 until multipart.getCount) {
          extractContentFromPart(multipart.getBodyPart(i))
        }
      case content: String =>
        elements += HTMLElement(
          ElementType.NARRATIVE_TEXT,
          content = content,
          metadata = recipientsMetadata)
      case _ =>
        elements += HTMLElement(
          ElementType.UNCATEGORIZED_TEXT,
          content = "Unknown content",
          metadata = recipientsMetadata)
    }

    elements.toArray
  }

  private def classifyMimeType(part: Part): String = {
    if (part.isMimeType("text/plain")) {
      MimeType.TEXT_PLAIN
    } else if (part.isMimeType("text/html")) {
      MimeType.TEXT_HTML
    } else if (part.isMimeType("multipart/*")) {
      MimeType.MULTIPART
    } else if (part.isMimeType("image/*")) {
      MimeType.IMAGE
    } else if (part.isMimeType("application/*")) {
      MimeType.APPLICATION
    } else {
      println(s"Unknown content type: ${part.getContentType}")
      MimeType.UNKNOWN
    }
  }

  private def getJavaMailSession = {
    val props = new Properties()
    props.put("mail.store.protocol", "smtp")
    val session = Session.getDefaultInstance(props, null)
    session
  }

  private def retrieveRecipients(mimeMessage: MimeMessage): mutable.Map[String, String] = {
    val from = mimeMessage.getFrom.mkString(", ")
    val to = mimeMessage.getRecipients(Message.RecipientType.TO).mkString(", ")
    val ccRecipients = mimeMessage.getRecipients(Message.RecipientType.CC)

    if (ccRecipients != null) {
      mutable.Map("sent_from" -> from, "sent_to" -> to, "cc_to" -> ccRecipients.mkString(", "))
    } else mutable.Map("sent_from" -> from, "sent_to" -> to)
  }

}
