package com.johnsnowlabs.reader

import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.flatspec.AnyFlatSpec

class TSVReaderTest extends AnyFlatSpec {

  val tsvFilesDirectory = "src/test/resources/reader/tsv"

  "TSVReader.tsv" should "include TABLE element in tsv array only if inferTableStructure=true" taggedAs FastTest in {
    val filePath = s"$tsvFilesDirectory/stanley-cups.tsv"

    val tsvReader = new TSVReader(inferTableStructure = true)
    val tsvDf = tsvReader.tsv(filePath)
    val elements = getFirstElementsArray(tsvDf, tsvReader.getOutputColumn)

    val tableElement = elements.find(_.elementType == ElementType.TABLE)
    val textElement = elements.find(_.elementType == ElementType.NARRATIVE_TEXT)

    assert(tableElement.isDefined)
    assert(tableElement.get.content.trim.nonEmpty)
    assert(textElement.isDefined)
    assert(textElement.get.content.trim.nonEmpty)
  }

  it should "produce normalized content without header by default" taggedAs FastTest in {
    val filePath = s"$tsvFilesDirectory/stanley-cups.tsv"

    val reader = new TSVReader(inferTableStructure = false)
    val tsvDf = reader.tsv(filePath)

    val elements = tsvDf.head.getAs[Seq[Row]]("tsv")
    val plainText = elements.head.getAs[String]("content")
    val expectedText = "Blues STL 1 Flyers PHI 2 Maple Leafs TOR 13"

    assert(cleanExtraWhitespace(plainText) == cleanExtraWhitespace(expectedText))
  }

  it should "use tsv as the default output column" taggedAs FastTest in {
    val reader = new TSVReader()

    assert(reader.getOutputColumn == "tsv")
  }

  private def getFirstElementsArray(df: DataFrame, outputCol: String): Array[HTMLElement] = {
    import df.sparkSession.implicits._
    df.select(outputCol).as[Array[HTMLElement]].head()
  }

  private def cleanExtraWhitespace(text: String): String = {
    text
      .replaceAll("[\\u00A0\\n]", " ")
      .replaceAll(" {2,}", " ")
      .trim
  }
}
