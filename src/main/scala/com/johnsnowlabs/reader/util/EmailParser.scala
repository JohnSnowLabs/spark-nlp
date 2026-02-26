package com.johnsnowlabs.reader.util

import com.johnsnowlabs.reader.MimeType
import jakarta.mail.internet.MimeMessage
import jakarta.mail.{Address, Message, Part}

import scala.collection.mutable

object EmailParser {

  private final val OLE2_MAGIC_BYTES = "D0CF11E0A1B11AE1"

  def isOutlookEmailFileType(bytes: Array[Byte]): Boolean = {
    if (bytes.length >= 8) {
      val header = bytes.take(8).map("%02X".format(_)).mkString
      header.startsWith(OLE2_MAGIC_BYTES)
    } else false
  }

  def classifyMimeType(part: Part): String = {
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

  def retrieveRecipients(mimeMessage: MimeMessage): mutable.Map[String, String] = {
    def safeMkString(addresses: Array[Address]): String =
      Option(addresses).map(_.mkString(", ")).getOrElse("")

    val from = safeMkString(mimeMessage.getFrom)
    val to = safeMkString(mimeMessage.getRecipients(Message.RecipientType.TO))
    val cc = safeMkString(mimeMessage.getRecipients(Message.RecipientType.CC))

    mutable.Map("sent_from" -> from, "sent_to" -> to, "cc_to" -> cc)
  }

}
