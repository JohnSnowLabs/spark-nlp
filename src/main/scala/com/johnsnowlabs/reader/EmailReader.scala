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
import com.johnsnowlabs.reader.util.EmailParser
import jakarta.mail._
import jakarta.mail.internet.MimeMessage
import org.apache.poi.hsmf.MAPIMessage
import org.apache.poi.hsmf.datatypes.AttachmentChunks
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
  * val emailsPath = "./email-files/test-several-attachments.eml"
  * val emailReader = new EmailReader()
  * val emailDf = emailReader.read(emailsPath)
  * }}}
  *
  * {{{
  * emailDf.show()
  * +--------------------+--------------------+
  * |                path|               email|
  * +--------------------+--------------------+
  * |file:/content/ema...|[{Title, Test Sev...|
  * +--------------------+--------------------+
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
  * For more examples please refer to this
  * [[https://github.com/JohnSnowLabs/spark-nlp/examples/python/reader/SparkNLP_Email_Reader_Demo.ipynb notebook]].
  */
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

  /** @param filePath
    *   this is a path to a directory of email files or a path to an email file E.g.
    *   "path/email/files"
    * @return
    *   Dataframe with parsed email content.
    */
  def email(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val lower = filePath.toLowerCase
      val baseDf = datasetWithBinaryFile(spark, filePath)

      val emailDf =
        if (lower.endsWith(".msg")) {
          baseDf.withColumn(outputColumn, parseOutlookEmailUDF(col("content")))
        } else {
          baseDf.withColumn(outputColumn, parseEmailUDF(col("content")))
        }

      if (storeContent) emailDf.select("path", outputColumn, "content")
      else emailDf.select("path", outputColumn)

    } else {
      throw new IllegalArgumentException(s"Invalid filePath: $filePath")
    }
  }

  private val parseEmailUDF = udf((data: Array[Byte]) => {
    val inputStream = new ByteArrayInputStream(data)
    parseEmailFile(inputStream)
  })

  private val parseOutlookEmailUDF = udf((data: Array[Byte]) => {
    val inputStream = new ByteArrayInputStream(data)
    parseMsgFile(inputStream)
  })

  def emailToHTMLElement(content: Array[Byte]): Seq[HTMLElement] = {
    val inputStream = new ByteArrayInputStream(content)
    try {
      if (EmailParser.isOutlookEmailFileType(content)) {
        parseMsgFile(inputStream)
      } else {
        parseEmailFile(inputStream)
      }
    } catch {
      case e: Exception =>
        Seq(
          HTMLElement(
            ElementType.ERROR,
            s"Could not parse email: ${e.getMessage}",
            mutable.Map()))
    } finally {
      inputStream.close()
    }
  }

  private def parseEmailFile(inputStream: InputStream): Array[HTMLElement] = {
    val session = getJavaMailSession
    val elements = ArrayBuffer[HTMLElement]()
    val mimeMessage = new MimeMessage(session, inputStream)

    val subject = mimeMessage.getSubject
    val recipientsMetadata = EmailParser.retrieveRecipients(mimeMessage)
    elements += HTMLElement(ElementType.TITLE, content = subject, metadata = recipientsMetadata)

    // Recursive function to process each part based on its type
    def extractContentFromPart(part: Part): Unit = {
      val partType = EmailParser.classifyMimeType(part)
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
        case MimeType.IMAGE =>
          val inputStream = part.getInputStream
          val bytes =
            Stream.continually(inputStream.read).takeWhile(_ != -1).map(_.toByte).toArray
          val base64Content = java.util.Base64.getEncoder.encodeToString(bytes)

          // Build metadata similar to HTMLReader <img> parsing
          val imgMetadata = mutable.Map[String, String]() ++ recipientsMetadata
          imgMetadata("encoding") = "base64"
          imgMetadata("fileName") = Option(part.getFileName).getOrElse("")
          imgMetadata("contentType") = part.getContentType

          // width/height usually not available in email headers, but we leave keys open
          elements += HTMLElement(
            ElementType.IMAGE,
            content = base64Content,
            metadata = imgMetadata)

        case MimeType.APPLICATION =>
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

  private def getJavaMailSession = {
    val props = new Properties()
    props.put("mail.store.protocol", "smtp")
    val session = Session.getDefaultInstance(props, null)
    session
  }

  private def parseMsgFile(inputStream: InputStream): Array[HTMLElement] = {
    val msg = new MAPIMessage(inputStream)
    val elements = ArrayBuffer[HTMLElement]()

    val recipientsMetadata = mutable.Map[String, String](
      "sent_from" -> Option(msg.getDisplayFrom).getOrElse(""),
      "sent_to" -> Option(msg.getDisplayTo).getOrElse(""))

    Option(msg.getSubject).foreach { subject =>
      elements += HTMLElement(ElementType.TITLE, content = subject, metadata = recipientsMetadata)
    }

    val body = Option(msg.getHtmlBody).orElse(Option(msg.getTextBody))
    body.foreach { b =>
      val mimeType = if (msg.getHtmlBody != null) "text/html" else "text/plain"
      elements += HTMLElement(
        ElementType.NARRATIVE_TEXT,
        content = b,
        metadata = recipientsMetadata ++ Map("mimeType" -> mimeType))
    }

    val attachments = msg.getAttachmentFiles
    if (attachments != null) {
      attachments.foreach { att: AttachmentChunks =>
        val name =
          Option(att.getAttachFileName)
            .map(_.toString)
            .orElse(Option(att.getAttachLongFileName).map(_.toString))
            .getOrElse("unnamed")

        val data = att.getAttachData
        if (data != null && data.getValue != null) {
          val bytes = data.getValue
          val base64Content = java.util.Base64.getEncoder.encodeToString(bytes)

          val contentType =
            Option(att.getAttachMimeTag).map(_.toString).getOrElse("application/octet-stream")

          if (contentType.toLowerCase.startsWith("image/")) {
            val imgMetadata = recipientsMetadata ++ Map(
              "encoding" -> "base64",
              "fileName" -> name,
              "contentType" -> contentType)
            elements += HTMLElement(
              ElementType.IMAGE,
              content = base64Content,
              metadata = imgMetadata)
          } else {
            elements += HTMLElement(
              ElementType.ATTACHMENT,
              content = name,
              metadata = recipientsMetadata ++ Map("contentType" -> contentType))
          }
        }
      }
    }

    elements.toArray
  }

}
