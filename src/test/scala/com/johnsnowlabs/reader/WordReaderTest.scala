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
import com.johnsnowlabs.reader.util.AssertReaders
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.{array_contains, col, explode, map_keys}
import org.scalatest.flatspec.AnyFlatSpec

class WordReaderTest extends AnyFlatSpec {

  private val spark = ResourceHelper.spark
  val docDirectory = "src/test/resources/reader/doc"

  import spark.implicits._

  "WordReader" should "read a directory of word files" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(docDirectory)

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
    assert(!wordDf.columns.contains("content"))
  }

  "WordReader" should "read a docx file with page breaks" taggedAs FastTest in {
    val wordReader = new WordReader(includePageBreaks = true)
    val wordDf = wordReader.doc(s"$docDirectory/page-breaks.docx")

    val pageBreakCount = wordDf
      .select(explode($"doc.metadata").as("metadata"))
      .filter(array_contains(map_keys($"metadata"), "pageBreak"))
      .count()

    assert(pageBreakCount == 5)
    assert(!wordDf.columns.contains("content"))
  }

  "WordReader" should "read a docx file with tables" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(s"$docDirectory/fake_table.docx")
    val htmlDf = wordDf
      .withColumn("doc_exploded", explode(col("doc")))
      .filter(col("doc_exploded.elementType") === "HTML")

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
    assert(!wordDf.columns.contains("content"))
    assert(htmlDf.count() == 0, "Expected no row with HTML element type")
  }

  "WordReader" should "read a docx file with images on it" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(s"$docDirectory/contains-pictures.docx")

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
    assert(!wordDf.columns.contains("content"))
  }

  "WordReader" should "store content" taggedAs FastTest in {
    val wordReader = new WordReader(storeContent = true)
    val wordDf = wordReader.doc(s"$docDirectory")

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
    assert(wordDf.columns.contains("content"))
  }

  it should "read docx file with tables as HTML form" taggedAs FastTest in {
    val wordReader = new WordReader(inferTableStructure = true, outputFormat = "html-table")
    val wordDf = wordReader.doc(s"$docDirectory/fake_table.docx")
    val htmlDf = wordDf
      .withColumn("doc_exploded", explode(col("doc")))
      .filter(col("doc_exploded.elementType") === ElementType.HTML)

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
    assert(htmlDf.count() > 0, "Expected at least one row with HTML element type")
  }

  it should "read docx file with tables as JSON form" taggedAs FastTest in {
    val wordReader = new WordReader(inferTableStructure = true)
    val wordDf = wordReader.doc(s"$docDirectory/fake_table.docx")
    val jsonDf = wordDf
      .withColumn("doc_exploded", explode(col("doc")))
      .filter(col("doc_exploded.elementType") === ElementType.JSON)

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
    assert(jsonDf.count() > 0, "Expected at least one row with JSON element type")
  }

  it should "read doc file with images on it" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(s"$docDirectory/contains-pictures.docx")
    val htmlDf = wordDf
      .withColumn("doc_exploded", explode(col("doc")))
      .filter(col("doc_exploded.elementType") === ElementType.IMAGE)

    assert(htmlDf.count() > 1)
  }

  it should "output hierarchy metadata" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(s"$docDirectory/hierarchy_test.docx")

    AssertReaders.assertHierarchy(wordDf, "doc")
  }

  it should "include domPath, orderTableIndex and orderImageIndex metadata fields" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(s"$docDirectory/doc-img-table.docx")

    val explodedDf = wordDf.withColumn("doc_exploded", explode(col("doc")))

    val tablesDf = explodedDf.filter(col("doc_exploded.elementType") === ElementType.TABLE)
    assert(tablesDf.count() > 0, "No TABLE elements found in WordReader output")

    val tableMetaDf = tablesDf.selectExpr(
      "doc_exploded.metadata.domPath as domPath",
      "doc_exploded.metadata.orderTableIndex as orderTableIndex")

    assert(
      tableMetaDf.filter(col("domPath").isNotNull).count() == tableMetaDf.count(),
      "Missing domPath in TABLE metadata")
    assert(
      tableMetaDf.filter(col("orderTableIndex").isNotNull).count() == tableMetaDf.count(),
      "Missing orderTableIndex in TABLE metadata")

    val imagesDf = explodedDf.filter(col("doc_exploded.elementType") === ElementType.IMAGE)
    assert(imagesDf.count() > 0, "No IMAGE elements found in WordReader output")

    val imageMetaDf = imagesDf.selectExpr(
      "doc_exploded.metadata.domPath as domPath",
      "doc_exploded.metadata.orderImageIndex as orderImageIndex")

    assert(
      imageMetaDf.filter(col("domPath").isNotNull).count() == imageMetaDf.count(),
      "Missing domPath in IMAGE metadata")
    assert(
      imageMetaDf.filter(col("orderImageIndex").isNotNull).count() == imageMetaDf.count(),
      "Missing orderImageIndex in IMAGE metadata")
  }

  it should "include coord field in IMAGE metadata with {x:...,y:...} format" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(s"$docDirectory/contains-pictures.docx")

    val explodedDf = wordDf.withColumn("doc_exploded", explode(col("doc")))
    val imagesDf = explodedDf.filter(col("doc_exploded.elementType") === ElementType.IMAGE)

    val coordDf = imagesDf.selectExpr("doc_exploded.metadata.coord as coord")

    assert(coordDf.count() > 0, "No coord field found in IMAGE metadata")

    val pattern = """\{x:\d+,y:\d+\}"""
    val allMatch = coordDf.collect().forall(row => row.getAs[String]("coord").matches(pattern))
    assert(allMatch, "Some IMAGE coord fields do not match expected {x:...,y:...} format")
  }

  it should "assign distinct coord values to each image" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(s"$docDirectory/contains-pictures.docx")

    val explodedDf = wordDf.withColumn("doc_exploded", explode(col("doc")))
    val imagesDf = explodedDf.filter(col("doc_exploded.elementType") === ElementType.IMAGE)
    val coords = imagesDf.selectExpr("doc_exploded.metadata.coord as coord").as[String].collect()

    assert(coords.nonEmpty, "No IMAGE coord metadata found")
    assert(coords.distinct.length == coords.length, "Duplicate IMAGE coordinates detected")
  }

}
