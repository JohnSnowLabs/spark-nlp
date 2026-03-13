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

import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.{col, explode}
import org.scalatest.flatspec.AnyFlatSpec

class EpubReaderTest extends AnyFlatSpec {

  private val epubDirectory = "src/test/resources/reader/epub"

  "EpubReader" should "read an epub file with text tables and images" taggedAs FastTest in {
    val epubReader = new EpubReader()
    val epubDf = epubReader.epub(s"$epubDirectory/sample.epub")
    val explodedDf = epubDf.withColumn("epub_exploded", explode(col("epub")))

    assert(epubDf.count() == 1)
    assert(explodedDf.filter(col("epub_exploded.elementType") === ElementType.TITLE).count() > 0)
    assert(
      explodedDf
        .filter(col("epub_exploded.elementType") === ElementType.NARRATIVE_TEXT)
        .count() > 0)
    assert(explodedDf.filter(col("epub_exploded.elementType") === ElementType.TABLE).count() > 0)

    val imageDf = explodedDf.filter(col("epub_exploded.elementType") === ElementType.IMAGE)
    assert(imageDf.count() > 0)
    assert(
      imageDf.filter(col("epub_exploded.binaryContent").isNotNull).count() == imageDf.count(),
      "Expected binaryContent for EPUB image elements")
  }

  it should "store raw file content when configured" taggedAs FastTest in {
    val epubReader = new EpubReader(storeContent = true)
    val epubDf = epubReader.epub(epubDirectory)

    assert(epubDf.columns.contains("content"))
  }
}
