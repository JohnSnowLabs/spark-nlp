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

}
