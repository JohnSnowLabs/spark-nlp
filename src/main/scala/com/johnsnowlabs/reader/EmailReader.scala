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
import jakarta.mail._
import jakarta.mail.internet.MimeMessage
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.io.{ByteArrayInputStream, InputStream}
import java.util.Properties
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/** This class is used to read and parse email content.
  *
  * @param addAttachmentContent
  *   Whether to extract and include the textual content of plain-text attachments in the output.
  *   By default, this is set to false.
  * @param storeContent
  *   Whether to include the raw file content in the output DataFrame as a separate 'content'
  *   column, alongside the structured output. By default, this is set to false.
  *
  * ==Example==
  * {{{
  * val emailsPath = "home/user/emails-directory"
  * val emailReader = new EmailReader()
  * val emailDf = emailReader.read(emailsPath)
  * }}}
  *
  * {{{
  * emailDf.select("email").show(truncate = false)
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |email                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[{Title, Email Text Attachments, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>}}, {NarrativeText, Email  test with two text attachments\r\n\r\nCheers,\r\n\r\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}, {NarrativeText, <html>\r\n<head>\r\n<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">\r\n<style type="text/css" style="display:none;"> P {margin-top:0;margin-bottom:0;} </style>\r\n</head>\r\n<body dir="ltr">\r\n<span style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">Email&nbsp; test with two text attachments</span>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\n<br>\r\n</div>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\nCheers,</div>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\n<br>\r\n</div>\r\n</body>\r\n</html>\r\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/html}}, {Attachment, filename.txt, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, contentType -> text/plain; name="filename.txt"}}, {NarrativeText, This is the content of the file.\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}, {Attachment, filename2.txt, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, contentType -> text/plain; name="filename2.txt"}}, {NarrativeText, This is an additional content file.\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}]|
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  *
  * emailDf.printSchema()
  * root
  *  |-- path: string (nullable = true)
  *  |-- email: array (nullable = true)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- elementType: string (nullable = true)
  *  |    |    |-- content: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  * }}}
  */

class EmailReader(addAttachmentContent: Boolean = false, storeContent: Boolean = false)
    extends Serializable {

  private val spark = ResourceHelper.spark
  import spark.implicits._

  def read(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val binaryFilesRDD = spark.sparkContext.binaryFiles(filePath)
      val byteArrayRDD = binaryFilesRDD.map { case (path, portableDataStream) =>
        val byteArray = portableDataStream.toArray()
        (path, byteArray)
      }
      val emailDf = byteArrayRDD
        .toDF("path", "content")
        .withColumn("email", parseEmailUDF(col("content")))
      if (storeContent) emailDf.select("path", "email", "content")
      else emailDf.select("path", "email")
    } else throw new IllegalArgumentException(s"Invalid filePath: $filePath")
  }

  private val parseEmailUDF = udf((data: Array[Byte]) => {
    val inputStream = new ByteArrayInputStream(data)
    parseEmailFile(inputStream)
  })

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
