package com.johnsnowlabs.reader.util

import com.johnsnowlabs.reader.MimeType
import jakarta.mail.internet.{InternetAddress, MimeBodyPart, MimeMessage}
import jakarta.mail.{Message, Session}
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.file.{Files, Paths}
import java.util.Properties
import scala.collection.mutable

class EmailParserTest extends AnyFlatSpec {

  "isOutlookEmailFileType" should "return true for a real Outlook .msg file" in {
    val path = Paths.get("src/test/resources/reader/email/email-test-image.msg")
    val bytes = Files.readAllBytes(path)

    assert(EmailParser.isOutlookEmailFileType(bytes))
  }

  it should "return false for .eml style (text header)" in {
    val emlBytes = "From: someone@example.com".getBytes("UTF-8")
    assert(!EmailParser.isOutlookEmailFileType(emlBytes))
  }

  it should "return false for insufficient bytes" in {
    assert(!EmailParser.isOutlookEmailFileType(Array(0xd0.toByte)))
  }

  "classifyMimeType" should "detect text/plain correctly" in {
    val part = new MimeBodyPart()
    part.setText("hello world", "utf-8", "plain")
    assert(EmailParser.classifyMimeType(part) == MimeType.TEXT_PLAIN)
  }

  it should "detect image/* correctly" in {
    val part = new MimeBodyPart()
    part.setHeader("Content-Type", "image/png")
    assert(EmailParser.classifyMimeType(part) == MimeType.IMAGE)
  }

  it should "detect application/* correctly" in {
    val part = new MimeBodyPart()
    part.setHeader("Content-Type", "application/pdf")
    assert(EmailParser.classifyMimeType(part) == MimeType.APPLICATION)
  }

  it should "fall back to UNKNOWN for unsupported mime" in {
    val part = new MimeBodyPart()
    part.setHeader("Content-Type", "weird/type")
    assert(EmailParser.classifyMimeType(part) == MimeType.UNKNOWN)
  }

  "retrieveRecipients" should "return correct recipients" in {
    val session = Session.getDefaultInstance(new Properties())
    val msg = new MimeMessage(session)
    msg.setFrom(new InternetAddress("from@example.com"))
    msg.setRecipients(Message.RecipientType.TO, "to@example.com")
    msg.setRecipients(Message.RecipientType.CC, "cc@example.com")

    val recipients: mutable.Map[String, String] = EmailParser.retrieveRecipients(msg)
    assert(recipients("sent_from").contains("from@example.com"))
    assert(recipients("sent_to").contains("to@example.com"))
    assert(recipients("cc_to").contains("cc@example.com"))
  }

  it should "handle null recipients gracefully" in {
    val session = Session.getDefaultInstance(new Properties())
    val msg = new MimeMessage(session)
    msg.setFrom(new InternetAddress("from@example.com"))

    val recipients: mutable.Map[String, String] = EmailParser.retrieveRecipients(msg)
    assert(recipients("sent_from").contains("from@example.com"))
    assert(recipients("sent_to") == "")
    assert(recipients("cc_to") == "")
  }
}
