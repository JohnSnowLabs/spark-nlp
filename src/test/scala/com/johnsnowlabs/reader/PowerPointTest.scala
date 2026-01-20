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
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.{col, explode}
import org.scalatest.flatspec.AnyFlatSpec

class PowerPointTest extends AnyFlatSpec {

  val docDirectory = "src/test/resources/reader/ppt"

  "PowerPointReader" should "read a power point file" taggedAs FastTest in {
    val powerPointReader = new PowerPointReader()
    val pptDf = powerPointReader.ppt(s"$docDirectory/fake-power-point.pptx")
    val narrativeTextDf = pptDf
      .withColumn("ppt_exploded", explode(col("ppt")))
      .filter(col("ppt_exploded.elementType") === ElementType.NARRATIVE_TEXT)

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
    assert(!pptDf.columns.contains("content"))
    assert(narrativeTextDf.count() == 2)
  }

  "PowerPointReader" should "read a power point directory" taggedAs FastTest in {
    val powerPointReader = new PowerPointReader()
    val pptDf = powerPointReader.ppt(s"$docDirectory")

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
    assert(!pptDf.columns.contains("content"))
  }

  "PowerPointReader" should "read a power point file with table" taggedAs FastTest in {
    val powerPointReader = new PowerPointReader()
    val pptDf = powerPointReader.ppt(s"$docDirectory/fake-power-point-table.pptx")

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
    assert(!pptDf.columns.contains("content"))
  }

  "PowerPointReader" should "store content" taggedAs FastTest in {
    val powerPointReader = new PowerPointReader(storeContent = true)
    val pptDf = powerPointReader.ppt(docDirectory)
    pptDf.show()

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
    assert(pptDf.columns.contains("content"))
  }

  it should "read speaker notes in a power point file" taggedAs FastTest in {
    val powerPointReader = new PowerPointReader(includeSlideNotes = true)
    val pptDf = powerPointReader.ppt(s"$docDirectory/speaker-notes.pptx")
    val narrativeTextDf = pptDf
      .withColumn("ppt_exploded", explode(col("ppt")))
      .filter(col("ppt_exploded.elementType") === ElementType.NARRATIVE_TEXT)

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
    assert(!pptDf.columns.contains("content"))
    assert(narrativeTextDf.count() == 3)
  }

  it should "read ppt file with tables as HTML form" taggedAs FastTest in {
    val powerPointReader =
      new PowerPointReader(inferTableStructure = true, outputFormat = "html-table")
    val pptDf = powerPointReader.ppt(s"$docDirectory/fake-power-point-table.pptx")
    val htmlDf = pptDf
      .withColumn("ppt_exploded", explode(col("ppt")))
      .filter(col("ppt_exploded.elementType") === ElementType.HTML)

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
    assert(htmlDf.count() > 0, "Expected at least one row with HTML element type")
  }

  it should "read ppt file with tables as JSON form" taggedAs FastTest in {
    val powerPointReader = new PowerPointReader(inferTableStructure = true)
    val pptDf = powerPointReader.ppt(s"$docDirectory/fake-power-point-table.pptx")
    val jsonDf = pptDf
      .withColumn("ppt_exploded", explode(col("ppt")))
      .filter(col("ppt_exploded.elementType") === ElementType.JSON)

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
    assert(jsonDf.count() > 0, "Expected at least one row with JSON element type")
  }

  it should "read images from ppt file" taggedAs FastTest in {
    val powerPointReader = new PowerPointReader()
    val pptDf = powerPointReader.ppt(s"$docDirectory/power-point-images.pptx")
    val imageDf = pptDf
      .withColumn("ppt_exploded", explode(col("ppt")))
      .filter(col("ppt_exploded.elementType") === ElementType.IMAGE)

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
    assert(imageDf.count() > 0, "Expected at least one row with IMAGE element type")
  }

  it should "include domPath and orderTableIndex metadata fields for tables" taggedAs FastTest in {
    val powerPointReader = new PowerPointReader()
    val pptDf = powerPointReader.ppt(s"$docDirectory/fake-power-point-table.pptx")

    val explodedDf = pptDf.withColumn("ppt_exploded", explode(col("ppt")))
    val tablesDf = explodedDf.filter(col("ppt_exploded.elementType") === ElementType.TABLE)

    assert(tablesDf.count() > 0, "No TABLE elements found in PowerPointReader output")

    val tableMetaDf = tablesDf.selectExpr(
      "ppt_exploded.metadata.domPath as domPath",
      "ppt_exploded.metadata.orderTableIndex as orderTableIndex")

    assert(
      tableMetaDf.filter(col("domPath").isNotNull).count() == tableMetaDf.count(),
      "Missing domPath in TABLE metadata")
    assert(
      tableMetaDf.filter(col("orderTableIndex").isNotNull).count() == tableMetaDf.count(),
      "Missing orderTableIndex in TABLE metadata")
  }

  it should "include domPath and orderImageIndex metadata fields for images" taggedAs FastTest in {
    val powerPointReader = new PowerPointReader()
    val pptDf = powerPointReader.ppt(s"$docDirectory/power-point-images.pptx")

    val explodedDf = pptDf.withColumn("ppt_exploded", explode(col("ppt")))
    val imagesDf = explodedDf.filter(col("ppt_exploded.elementType") === ElementType.IMAGE)

    assert(imagesDf.count() > 0, "No IMAGE elements found in PowerPointReader output")

    val imageMetaDf = imagesDf.selectExpr(
      "ppt_exploded.metadata.domPath as domPath",
      "ppt_exploded.metadata.orderImageIndex as orderImageIndex")

    assert(
      imageMetaDf.filter(col("domPath").isNotNull).count() == imageMetaDf.count(),
      "Missing domPath in IMAGE metadata")
    assert(
      imageMetaDf.filter(col("orderImageIndex").isNotNull).count() == imageMetaDf.count(),
      "Missing orderImageIndex in IMAGE metadata")
  }

  it should "include coord field in IMAGE metadata with {x:...,y:...} format" taggedAs FastTest in {
    val powerPointReader = new PowerPointReader()
    val pptDf = powerPointReader.ppt(s"$docDirectory/power-point-images.pptx")

    val explodedDf = pptDf.withColumn("ppt_exploded", explode(col("ppt")))
    val imageDf = explodedDf.filter(col("ppt_exploded.elementType") === ElementType.IMAGE)

    assert(imageDf.count() > 0, "No IMAGE elements found in PowerPointReader output")

    val coordDf = imageDf.selectExpr("ppt_exploded.metadata.coord as coord")

    assert(coordDf.count() > 0, "Expected coord metadata field for IMAGE elements")

    val pattern = """\{x:\d+,y:\d+\}"""
    val allMatch = coordDf.collect().forall(row => row.getAs[String]("coord").matches(pattern))
    assert(allMatch, "Some IMAGE coord fields do not match the expected {x:...,y:...} format")
  }

  it should "assign distinct coord values to images across slides" taggedAs FastTest in {
    val spark = ResourceHelper.spark
    import spark.implicits._

    val powerPointReader = new PowerPointReader()
    val pptDf = powerPointReader.ppt(s"$docDirectory/power-point-images.pptx")

    val explodedDf = pptDf.withColumn("ppt_exploded", explode(col("ppt")))
    val imageDf = explodedDf.filter(col("ppt_exploded.elementType") === ElementType.IMAGE)

    val coords = imageDf.selectExpr("ppt_exploded.metadata.coord as coord").as[String].collect()

    assert(coords.nonEmpty, "No IMAGE coord metadata found")
    assert(
      coords.distinct.length == coords.length,
      "Duplicate IMAGE coordinates detected across slides")
  }

  it should "include slide and image indices in domPath for images" taggedAs FastTest in {
    val spark = ResourceHelper.spark
    import spark.implicits._

    val powerPointReader = new PowerPointReader()
    val pptDf = powerPointReader.ppt(s"$docDirectory/power-point-images.pptx")

    val explodedDf = pptDf.withColumn("ppt_exploded", explode(col("ppt")))
    val imagesDf = explodedDf.filter(col("ppt_exploded.elementType") === ElementType.IMAGE)

    val domPaths =
      imagesDf.selectExpr("ppt_exploded.metadata.domPath as domPath").as[String].collect()
    assert(
      domPaths.forall(_.matches(".*/slide\\[\\d+\\]/image\\[\\d+\\]")),
      "Invalid domPath structure in IMAGE metadata")
  }

}
