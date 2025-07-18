/*
 *   Copyright 2017-2024 John Snow Labs
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

import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.{col, explode}
import org.scalatest.flatspec.AnyFlatSpec

class HTMLReaderTest extends AnyFlatSpec {

  val htmlFilesDirectory = "./src/test/resources/reader/html/"

  it should "read html as dataframe" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(htmlFilesDirectory)
    htmlDF.show()

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(!htmlDF.columns.contains("content"))
  }

  it should "read html as dataframe with params" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(titleFontSize = 12)
    val htmlDF = HTMLReader.read(htmlFilesDirectory)
    htmlDF.show()

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(!htmlDF.columns.contains("content"))
  }

  it should "parse an html in real time" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read("https://www.wikipedia.org")
    htmlDF.show()

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(!htmlDF.columns.contains("content"))
  }

  it should "parse URLS in real time" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(Array("https://www.wikipedia.org", "https://example.com/"))
    htmlDF.show()

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(!htmlDF.columns.contains("content"))
  }

  it should "store content" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(storeContent = true)
    val htmlDF = HTMLReader.read(htmlFilesDirectory)
    htmlDF.show()

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
    assert(htmlDF.columns.contains("content"))
  }

  it should "work with headers" taggedAs FastTest in {
    val HTMLReader =
      new HTMLReader(headers = Map("User-Agent" -> "Mozilla/5.0", "Accept-Language" -> "es-ES"))
    val htmlDF = HTMLReader.read("https://www.google.com")
    htmlDF.show()

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
    htmlDF.show(truncate = false)

    val titleDF = htmlDF
      .select(explode(col("html")).as("exploded_html"))
      .filter(col("exploded_html.elementType") === ElementType.TITLE)

    assert(titleDF.count() == 1)
  }

}
