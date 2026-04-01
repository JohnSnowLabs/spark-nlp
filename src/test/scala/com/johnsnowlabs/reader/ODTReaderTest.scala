/*
 * Copyright 2017-2026 John Snow Labs
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
import com.johnsnowlabs.reader.util.AssertReaders
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.{array_contains, col, explode, map_keys}
import org.scalatest.flatspec.AnyFlatSpec

class ODTReaderTest extends AnyFlatSpec {

  private val spark = ResourceHelper.spark
  val odtDirectory = "src/test/resources/reader/odt"

  import spark.implicits._

  "ODTReader" should "read a directory of odt files" taggedAs FastTest in {
    val odtReader = new ODTReader()
    val odtDf = odtReader.doc(odtDirectory)

    assert(!odtDf.select(col("doc").getItem(0)).isEmpty)
    assert(!odtDf.columns.contains("content"))
  }
  it should "read an odt file with page breaks" taggedAs FastTest in {
    val odtReader = new ODTReader(includePageBreaks = true)
    val odtDf = odtReader.doc(s"$odtDirectory/page-breaks.odt")

    val pageBreakCount = odtDf
      .withColumn("doc_exploded", explode(col("doc")))
      .filter(array_contains(map_keys(col("doc_exploded.metadata")), "pageBreak"))
      .count()

    assert(pageBreakCount == 3)
    assert(!odtDf.columns.contains("content"))
  }

  it should "read an odt file with tables" taggedAs FastTest in {
    val odtReader = new ODTReader()
    val odtDf = odtReader.doc(s"$odtDirectory/fake_table.odt")
    val htmlDf = odtDf
      .withColumn("doc_exploded", explode(col("doc")))
      .filter(col("doc_exploded.elementType") === ElementType.HTML)

    assert(!odtDf.select(col("doc").getItem(0)).isEmpty)
    assert(!odtDf.columns.contains("content"))
    assert(htmlDf.count() == 0, "Expected no row with HTML element type")
  }

  it should "read an odt file with images on it" taggedAs FastTest in {
    val odtReader = new ODTReader()
    val odtDf = odtReader.doc(s"$odtDirectory/contains-pictures.odt")

    assert(!odtDf.select(col("doc").getItem(0)).isEmpty)
    assert(!odtDf.columns.contains("content"))
  }

  it should "store content" taggedAs FastTest in {
    val odtReader = new ODTReader(storeContent = true)
    val odtDf = odtReader.doc(odtDirectory)

    assert(!odtDf.select(col("doc").getItem(0)).isEmpty)
    assert(odtDf.columns.contains("content"))
  }

  it should "read odt tables as HTML form" taggedAs FastTest in {
    val odtReader = new ODTReader(inferTableStructure = true, outputFormat = "html-table")
    val odtDf = odtReader.doc(s"$odtDirectory/fake_table.odt")
    val htmlDf = odtDf
      .withColumn("doc_exploded", explode(col("doc")))
      .filter(col("doc_exploded.elementType") === ElementType.HTML)

    assert(!odtDf.select(col("doc").getItem(0)).isEmpty)
    assert(htmlDf.count() > 0, "Expected at least one row with HTML element type")
  }

  it should "read odt tables as JSON form" taggedAs FastTest in {
    val odtReader = new ODTReader(inferTableStructure = true)
    val odtDf = odtReader.doc(s"$odtDirectory/fake_table.odt")
    val jsonDf = odtDf
      .withColumn("doc_exploded", explode(col("doc")))
      .filter(col("doc_exploded.elementType") === ElementType.JSON)

    assert(!odtDf.select(col("doc").getItem(0)).isEmpty)
    assert(jsonDf.count() > 0, "Expected at least one row with JSON element type")
  }

  it should "output hierarchy metadata" taggedAs FastTest in {
    val odtReader = new ODTReader()
    val odtDf = odtReader.doc(s"$odtDirectory/hierarchy_test.odt")

    AssertReaders.assertHierarchy(odtDf, "doc")
  }

  it should "include paragraph_index and paragraph_y metadata fields for text elements" taggedAs FastTest in {
    val paragraphSpacingY = 25
    val odtReader = new ODTReader()
    val odtDf = odtReader.doc(s"$odtDirectory/hierarchy_test.odt")

    val textDf = odtDf
      .withColumn("doc_exploded", explode(col("doc")))
      .filter(
        col("doc_exploded.elementType")
          .isin(
            ElementType.TITLE,
            ElementType.NARRATIVE_TEXT,
            ElementType.LIST_ITEM,
            ElementType.TABLE))
      .selectExpr(
        "cast(doc_exploded.metadata.paragraph_index as int) as paragraphIndex",
        "cast(doc_exploded.metadata.paragraph_y as int) as paragraphY")

    assert(textDf.count() > 0, "No text elements found in ODTReader output")
    assert(textDf.filter(col("paragraphIndex").isNull).count() == 0)
    assert(textDf.filter(col("paragraphY").isNull).count() == 0)
    assert(
      textDf.filter(col("paragraphY") =!= col("paragraphIndex") * paragraphSpacingY).count() == 0,
      "paragraph_y should be derived from paragraph_index")
  }

  it should "include domPath, orderTableIndex and orderImageIndex metadata fields" taggedAs FastTest in {
    val odtReader = new ODTReader()
    val odtDf = odtReader.doc(s"$odtDirectory/doc-img-table.odt")

    val explodedDf = odtDf.withColumn("doc_exploded", explode(col("doc")))

    val tablesDf = explodedDf.filter(col("doc_exploded.elementType") === ElementType.TABLE)
    assert(tablesDf.count() > 0, "No TABLE elements found in ODTReader output")

    val tableMetaDf = tablesDf.selectExpr(
      "doc_exploded.metadata.domPath as domPath",
      "doc_exploded.metadata.orderTableIndex as orderTableIndex")

    assert(tableMetaDf.filter(col("domPath").isNotNull).count() == tableMetaDf.count())
    assert(tableMetaDf.filter(col("orderTableIndex").isNotNull).count() == tableMetaDf.count())

    val imagesDf = explodedDf.filter(col("doc_exploded.elementType") === ElementType.IMAGE)
    assert(imagesDf.count() > 0, "No IMAGE elements found in ODTReader output")

    val imageMetaDf = imagesDf.selectExpr(
      "doc_exploded.metadata.domPath as domPath",
      "doc_exploded.metadata.orderImageIndex as orderImageIndex")

    assert(imageMetaDf.filter(col("domPath").isNotNull).count() == imageMetaDf.count())
    assert(imageMetaDf.filter(col("orderImageIndex").isNotNull).count() == imageMetaDf.count())
  }

  it should "include coord field in IMAGE metadata with {x:...,y:...} format" taggedAs FastTest in {
    val odtReader = new ODTReader()
    val odtDf = odtReader.doc(s"$odtDirectory/contains-pictures.odt")

    val explodedDf = odtDf.withColumn("doc_exploded", explode(col("doc")))
    val imagesDf = explodedDf.filter(col("doc_exploded.elementType") === ElementType.IMAGE)
    val coordDf = imagesDf.selectExpr("doc_exploded.metadata.coord as coord")

    assert(coordDf.count() > 0, "No coord field found in IMAGE metadata")

    val pattern = """\{x:\d+,y:\d+\}"""
    val allMatch = coordDf.collect().forall(row => row.getAs[String]("coord").matches(pattern))
    assert(allMatch, "Some IMAGE coord fields do not match expected {x:...,y:...} format")
  }

  it should "assign distinct coord values to each image" taggedAs FastTest in {
    val odtReader = new ODTReader()
    val odtDf = odtReader.doc(s"$odtDirectory/contains-pictures.odt")

    val explodedDf = odtDf.withColumn("doc_exploded", explode(col("doc")))
    val imagesDf = explodedDf.filter(col("doc_exploded.elementType") === ElementType.IMAGE)
    val coords = imagesDf.selectExpr("doc_exploded.metadata.coord as coord").as[String].collect()

    assert(coords.nonEmpty, "No IMAGE coord metadata found")
    assert(coords.distinct.length == coords.length, "Duplicate IMAGE coordinates detected")
  }

  it should "tag inline and floating images in metadata" taggedAs FastTest in {
    val odtReader = new ODTReader()
    val odtDf = odtReader.doc(s"$odtDirectory/contains-pictures.odt")

    val imagesDf = odtDf
      .withColumn("doc_exploded", explode(col("doc")))
      .filter(col("doc_exploded.elementType") === ElementType.IMAGE)

    val imageTypes = imagesDf
      .selectExpr("doc_exploded.metadata.image_type as image_type")
      .as[String]
      .collect()
      .toSet

    assert(imageTypes.contains("inline"), "Missing inline image_type in IMAGE metadata")
    assert(imageTypes.contains("floating"), "Missing floating image_type in IMAGE metadata")
  }
}
