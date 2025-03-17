package com.johnsnowlabs.partition

import org.apache.spark.sql.functions.col
import org.scalatest.flatspec.AnyFlatSpec

class PartitionTest extends AnyFlatSpec {

  val txtDirectory = "src/test/resources/reader/txt"
  val wordDirectory = "src/test/resources/reader/doc"
  val excelDirectory = "src/test/resources/reader/xls"
  val powerPointDirectory = "src/test/resources/reader/ppt"
  val emailDirectory = "src/test/resources/reader/email"
  val htmlDirectory = "src/test/resources/reader/html"

  "Partition" should "work with text content_type" in {
    val textDf = Partition(Map("content_type" -> "text/plain")).partition(txtDirectory)
    textDf.show()

    assert(!textDf.select(col("txt").getItem(0)).isEmpty)
  }

  it should "identify text file" in {
    val textDf = Partition().partition(s"$txtDirectory/simple-text.txt")
    textDf.show()

    assert(!textDf.select(col("txt").getItem(0)).isEmpty)
  }

  it should "work with word content_type" in {
    val wordDf = Partition(Map("content_type" -> "application/msword")).partition(wordDirectory)
    wordDf.show()

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
  }

  it should "identify word file" in {
    val wordDf = Partition().partition(s"$wordDirectory/fake_table.docx")
    wordDf.show()

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
  }

  it should "work with excel content_type" in {
    val excelDf =
      Partition(Map("content_type" -> "application/vnd.ms-excel")).partition(excelDirectory)
    excelDf.show()

    assert(!excelDf.select(col("xls").getItem(0)).isEmpty)
  }

  it should "identify excel file" in {
    val excelDf = Partition().partition(s"$excelDirectory/vodafone.xlsx")
    excelDf.show()

    assert(!excelDf.select(col("xls").getItem(0)).isEmpty)
  }

  it should "work with email content_type" in {
    val emailDf = Partition(Map("content_type" -> "message/rfc822")).partition(emailDirectory)
    emailDf.show()

    assert(!emailDf.select(col("email").getItem(0)).isEmpty)
  }

  it should "wok with email file" in {
    val emailDf = Partition().partition(s"$emailDirectory/test-several-attachments.eml")
    emailDf.show()

    assert(!emailDf.select(col("email").getItem(0)).isEmpty)
  }

  it should "work with powerpoint content_type" in {
    val pptDf = Partition(Map("content_type" -> "application/vnd.ms-powerpoint"))
      .partition(powerPointDirectory)
    pptDf.show()

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
  }

  it should "identify powerpoint file" in {
    val pptDf = Partition().partition(s"$powerPointDirectory/fake-power-point.pptx")
    pptDf.show()

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
  }

  it should "work with html content_type" in {
    val htmlDf = Partition(Map("content_type" -> "text/html")).partition(htmlDirectory)
    htmlDf.show()

    assert(!htmlDf.select(col("html").getItem(0)).isEmpty)
  }

  it should "identify html file" in {
    val htmlDf = Partition().partition(s"$htmlDirectory/fake-html.html")
    htmlDf.show()

    assert(!htmlDf.select(col("html").getItem(0)).isEmpty)
  }

  it should "work with an URL" in {
    val htmlDf = Partition().partition("https://www.wikipedia.org")
    htmlDf.show()

    assert(!htmlDf.select(col("html").getItem(0)).isEmpty)
  }

  it should "work with a set of URLS" in {
    val htmlDf = Partition().partition(Array("https://www.wikipedia.org", "https://example.com/"))
    htmlDf.show()

    assert(!htmlDf.select(col("html").getItem(0)).isEmpty)
  }

}
