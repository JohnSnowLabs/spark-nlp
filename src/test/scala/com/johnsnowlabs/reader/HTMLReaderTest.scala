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
    htmlDF.show(truncate = false)
    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TITLE)
    titleDF.select("exploded_html").show(truncate = false)

    assert(titleDF.count() == 2)
  }

  it should "output as title for font size >= 22" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(titleFontSize = 22)

    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/title-test.html")
    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TITLE)
    titleDF.select("exploded_html").show(truncate = false)

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
    htmlDF.show(truncate = false)
    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TITLE)

    assert(titleDF.count() == 2)
  }

  it should "correctly parse caption and th tags" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/example-caption-th.html")
    htmlDF.show(truncate = false)
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

  it should "include parent and element ids" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/simple-book.html")
    htmlDF.show(truncate = false)
    val parentChildDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))

    parentChildDF.show(truncate = false)

//    assert(parentChildDF.count() == 3)
  }

  it should "produce valid element_id and parent_id relationships" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(s"$htmlFilesDirectory/simple-book.html")

    val explodedDF = htmlDF
      .select(explode(col("html")).as("elem"))
      .select(
        col("elem.elementType").as("elementType"),
        col("elem.content").as("content"),
        col("elem.metadata").as("metadata"))
      .withColumn("element_id", col("metadata")("element_id"))
      .withColumn("parent_id", col("metadata")("parent_id"))
      .cache() // << important to prevent recomputation inconsistencies

    val allElementIds = explodedDF
      .select("element_id")
      .where(col("element_id").isNotNull)
      .distinct()
      .collect()
      .map(_.getString(0))
      .toSet

    val allParentIds = explodedDF
      .select("parent_id")
      .where(col("parent_id").isNotNull)
      .distinct()
      .collect()
      .map(_.getString(0))
      .toSet

    // 1. There should be at least one element with an element_id
    assert(allElementIds.nonEmpty, "No elements have element_id metadata")

    // 2. There should be at least one element with a parent_id
    assert(allParentIds.nonEmpty, "No elements have parent_id metadata")

    // 3. Every parent_id should exist as an element_id
    val missingParents = allParentIds.diff(allElementIds)
    assert(
      missingParents.isEmpty,
      s"Some parent_ids do not correspond to existing element_ids: $missingParents")

    // 4. Each parent should have at least one child
    val parentChildCount = explodedDF
      .filter(col("parent_id").isNotNull)
      .groupBy("parent_id")
      .count()
      .collect()
      .map(r => r.getString(0) -> r.getLong(1))
      .toMap

    assert(
      parentChildCount.nonEmpty && parentChildCount.values.forall(_ >= 1),
      "Each parent_id should have at least one child element")
  }

}
