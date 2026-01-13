/*
 *   Copyright 2017-2025 John Snow Labs
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

import com.johnsnowlabs.reader.util.AssertReaders
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.sql.functions.{col, explode}
import org.scalatest.flatspec.AnyFlatSpec

class HTMLReaderTest extends AnyFlatSpec {

  val htmlFilesDirectory = "./src/test/resources/reader/html/"

  it should "read html as dataframe" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(htmlFilesDirectory)

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(!htmlDF.columns.contains("content"))
  }

  it should "read html as dataframe with params" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(titleFontSize = 12)
    val htmlDF = HTMLReader.read(htmlFilesDirectory)

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(!htmlDF.columns.contains("content"))
  }

  it should "parse an html in real time" taggedAs SlowTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read("https://www.wikipedia.org")

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(!htmlDF.columns.contains("content"))
  }

  it should "parse URLS in real time" taggedAs SlowTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(Array("https://www.wikipedia.org", "https://example.com/"))

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(!htmlDF.columns.contains("content"))
  }

  it should "store content" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(storeContent = true)
    val htmlDF = HTMLReader.read(htmlFilesDirectory)

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(htmlDF.columns.contains("content"))
  }

  it should "work with headers" taggedAs FastTest in {
    val HTMLReader =
      new HTMLReader(headers = Map("User-Agent" -> "Mozilla/5.0", "Accept-Language" -> "es-ES"))
    val htmlDF = HTMLReader.read("https://www.google.com")

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(!htmlDF.columns.contains("content"))
  }

  it should "output as title for font size >= 19" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(titleFontSize = 19)

    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/title-test.html")

    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TITLE)

    assert(titleDF.count() == 2)
  }

  it should "output as title for font size >= 22" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(titleFontSize = 22)

    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/title-test.html")
    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TITLE)

    assert(titleDF.count() == 1)
  }

  it should "correctly parse div tags" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/example-div.html")
    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TITLE)
    val textDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.NARRATIVE_TEXT)

    assert(titleDF.count() == 1)
    assert(textDF.count() == 1)
  }

  it should "correctly parse bold and strong tags" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/example-bold-strong.html")

    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TITLE)

    assert(titleDF.count() == 2)
  }

  it should "correctly parse caption and th tags" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/example-caption-th.html")

    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TABLE)

    assert(titleDF.count() == 1)
  }

  it should "include title tag value in metadata" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(includeTitleTag = true)
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/example-caption-th.html")

    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TITLE)

    assert(titleDF.count() == 1)
  }

  it should "output table JSON" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(outputFormat = "json-table")
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/example-caption-th.html")
    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TABLE)

    assert(titleDF.count() == 1)
  }

  it should "output table as HTML" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(outputFormat = "html-table")
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/example-caption-th.html")
    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TABLE)

    assert(titleDF.count() == 1)
  }

  it should "read HTML files with images" taggedAs SlowTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/example-images.html")

    val imagesDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.IMAGE)

    assert(imagesDF.count() == 3)
  }

  it should "read HTML files with images inside paragraphs" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/example-image-paragraph.html")

    val imagesDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.IMAGE)

    assert(imagesDF.count() == 1)
  }

  it should "produce valid element_id and parent_id relationships" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/simple-book.html")

    AssertReaders.assertHierarchy(htmlDF, "html")
  }

  it should "include domPath and orderTableIndex metadata fields for tables" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/sample_tables.html")

    val explodedDf = htmlDF.withColumn("html_exploded", explode(col("html")))
    val tablesDf = explodedDf.filter(col("html_exploded.elementType") === ElementType.TABLE)

    assert(tablesDf.count() > 0, "No TABLE elements found in HTMLReader output")

    val tableMetaDf = tablesDf.selectExpr(
      "html_exploded.metadata.domPath as domPath",
      "html_exploded.metadata.orderTableIndex as orderTableIndex")

    assert(
      tableMetaDf.filter(col("domPath").isNotNull).count() == tableMetaDf.count(),
      "Missing domPath in TABLE metadata")
    assert(
      tableMetaDf.filter(col("orderTableIndex").isNotNull).count() == tableMetaDf.count(),
      "Missing orderTableIndex in TABLE metadata")
  }

  it should "include domPath and orderImageIndex metadata fields for images" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/sample_images.html")

    val explodedDf = htmlDF.withColumn("html_exploded", explode(col("html")))
    val imagesDf = explodedDf.filter(col("html_exploded.elementType") === ElementType.IMAGE)

    assert(imagesDf.count() > 0, "No IMAGE elements found in HTMLReader output")

    val imageMetaDf = imagesDf.selectExpr(
      "html_exploded.metadata.domPath as domPath",
      "html_exploded.metadata.orderImageIndex as orderImageIndex")

    assert(
      imageMetaDf.filter(col("domPath").isNotNull).count() == imageMetaDf.count(),
      "Missing domPath in IMAGE metadata")
    assert(
      imageMetaDf.filter(col("orderImageIndex").isNotNull).count() == imageMetaDf.count(),
      "Missing orderImageIndex in IMAGE metadata")
  }

  it should "include domPath, orderTableIndex and orderImageIndex metadata fields for tables and images" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/sample_mixed.html")

    val explodedDf = htmlDF.withColumn("html_exploded", explode(col("html")))

    val tablesDf = explodedDf.filter(col("html_exploded.elementType") === ElementType.TABLE)
    assert(tablesDf.count() > 0, "No TABLE elements found in mixed HTML output")

    val tableMetaDf = tablesDf.selectExpr(
      "html_exploded.metadata.domPath as domPath",
      "html_exploded.metadata.orderTableIndex as orderTableIndex")
    assert(
      tableMetaDf.filter(col("domPath").isNotNull).count() == tableMetaDf.count(),
      "Missing domPath in TABLE metadata")
    assert(
      tableMetaDf.filter(col("orderTableIndex").isNotNull).count() == tableMetaDf.count(),
      "Missing orderTableIndex in TABLE metadata")

    val imagesDf = explodedDf.filter(col("html_exploded.elementType") === ElementType.IMAGE)
    assert(imagesDf.count() > 0, "No IMAGE elements found in mixed HTML output")

    val imageMetaDf = imagesDf.selectExpr(
      "html_exploded.metadata.domPath as domPath",
      "html_exploded.metadata.orderImageIndex as orderImageIndex")
    assert(
      imageMetaDf.filter(col("domPath").isNotNull).count() == imageMetaDf.count(),
      "Missing domPath in IMAGE metadata")
    assert(
      imageMetaDf.filter(col("orderImageIndex").isNotNull).count() == imageMetaDf.count(),
      "Missing orderImageIndex in IMAGE metadata")
  }

  it should "include x_coordinate and y_coordinate metadata fields for images" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/example-image-coordinates.html")

    val explodedDf = htmlDF.withColumn("html_exploded", explode(col("html")))
    val imagesDf = explodedDf.filter(col("html_exploded.elementType") === ElementType.IMAGE)

    assert(imagesDf.count() == 2, "Expected exactly two images in test HTML")

    val imageMetaDf = imagesDf.selectExpr(
      "html_exploded.metadata.x_coordinate as x_coordinate",
      "html_exploded.metadata.y_coordinate as y_coordinate")

    // Both coordinates should exist, either from CSS or from fallback heuristic
    assert(
      imageMetaDf.filter(col("x_coordinate").isNotNull).count() == imageMetaDf.count(),
      "Missing x_coordinate in IMAGE metadata")
    assert(
      imageMetaDf.filter(col("y_coordinate").isNotNull).count() == imageMetaDf.count(),
      "Missing y_coordinate in IMAGE metadata")

    assert(
      imageMetaDf.filter(col("x_coordinate").rlike("^[0-9]+$")).count() == imageMetaDf.count())

  }

}
