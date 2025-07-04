package com.johnsnowlabs.reader

import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.flatspec.AnyFlatSpec

class CSVReaderTest extends AnyFlatSpec {

  val cvsFilesDirectory = "src/test/resources/reader/csv"

  "CSVReader.csv" should "include TABLE element in csv array only if inferTableStructure=true" in {
    val filePath = s"$cvsFilesDirectory/stanley-cups.csv"

    val csvReader = new CSVReader(inferTableStructure = true)
    val csvDf = csvReader.csv(filePath)
    val elementsTrue = getFirstElementsArray(csvDf, csvReader.getOutputColumn)

    val tableElement = elementsTrue.find(_.elementType == ElementType.TABLE)
    val textElement = elementsTrue.find(_.elementType == ElementType.NARRATIVE_TEXT)

    assert(tableElement.isDefined)
    assert(tableElement.get.content.trim.nonEmpty)
    assert(textElement.isDefined)
    assert(textElement.get.content.trim.nonEmpty)
  }

  "CSVReader.csv" should "include only text element in csv array if inferTableStructure=false" in {
    val filePath = s"$cvsFilesDirectory/stanley-cups.csv"

    val csvReader = new CSVReader(inferTableStructure = false)
    val csvDf = csvReader.csv(filePath)

    val elementsTrue = getFirstElementsArray(csvDf, csvReader.getOutputColumn)

    val tableElement = elementsTrue.find(_.elementType == ElementType.TABLE)
    val textElement = elementsTrue.find(_.elementType == ElementType.NARRATIVE_TEXT)

    assert(tableElement.isEmpty)
    assert(textElement.isDefined)
    assert(textElement.get.content.trim.nonEmpty)
  }

  def getFirstElementsArray(df: DataFrame, outputCol: String): Array[HTMLElement] = {
    import df.sparkSession.implicits._
    df.select(outputCol).as[Array[HTMLElement]].head()
  }

  "CSVReader" should "produce normalized content including header when includeHeader = true" in {
    val filePath = s"$cvsFilesDirectory/stanley-cups-utf-16.csv"

    val reader = new CSVReader(encoding = "UTF-16", includeHeader = true)
    val csvDf = reader.csv(filePath)

    val elements = csvDf.head.getAs[Seq[Row]]("csv")
    val plainText = elements.head.getAs[String]("content")
    val EXPECTED_TEXT =
      "Stanley Cups Team Location Stanley Cups Blues STL 1 Flyers PHI 2 Maple Leafs TOR 13"
    val result = cleanExtraWhitespace(plainText)
    val expected = cleanExtraWhitespace(EXPECTED_TEXT)

    assert(result == expected)
  }

  "CSVReader" should "produce normalized content without header when includeHeader = false" in {
    val filePath = s"$cvsFilesDirectory/stanley-cups-utf-16.csv"

    val reader = new CSVReader(encoding = "UTF-16", includeHeader = false)
    val csvDf = reader.csv(filePath)

    val elements = csvDf.head.getAs[Seq[Row]]("csv")
    val plainText = elements.head.getAs[String]("content")
    val EXPECTED_TEXT = "Team Location Stanley Cups Blues STL 1 Flyers PHI 2 Maple Leafs TOR 13"
    val result = cleanExtraWhitespace(plainText)
    val expected = cleanExtraWhitespace(EXPECTED_TEXT)

    assert(result == expected)
  }

  "CSVReader.csv" should "work for other delimiters" in {
    val filePath = s"$cvsFilesDirectory/semicolon-delimited.csv"

    val csvReader = new CSVReader(inferTableStructure = false, delimiter = ";")
    val csvDf = csvReader.csv(filePath)
    val elementsTrue = getFirstElementsArray(csvDf, csvReader.getOutputColumn)

    val tableElement = elementsTrue.find(_.elementType == ElementType.TABLE)
    val textElement = elementsTrue.find(_.elementType == ElementType.NARRATIVE_TEXT)

    assert(tableElement.isEmpty)
    assert(textElement.isDefined)
    assert(textElement.get.content.trim.nonEmpty)
  }

  def getFirstElement(df: org.apache.spark.sql.DataFrame, outputCol: String): HTMLElement = {
    import df.sparkSession.implicits._
    df.select(outputCol).as[Seq[HTMLElement]].head.head
  }

  def cleanExtraWhitespace(text: String): String = {
    // Replace non-breaking spaces (\u00A0) and newlines with space
    val cleanedText = text
      .replaceAll("[\\u00A0\\n]", " ")
      .replaceAll(" {2,}", " ")
    cleanedText.trim
  }

}
