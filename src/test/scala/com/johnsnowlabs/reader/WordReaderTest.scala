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
import org.apache.spark.sql.functions.{array_contains, col, explode, map_keys}
import org.scalatest.flatspec.AnyFlatSpec

class WordReaderTest extends AnyFlatSpec {

  private val spark = ResourceHelper.spark
  val docDirectory = "src/test/resources/reader/doc"

  import spark.implicits._

  "WordReader" should "read a directory of word files" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(docDirectory)
    wordDf.select("doc").show(false)
    wordDf.printSchema()
    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
    assert(!wordDf.columns.contains("content"))
  }

  "WordReader" should "read a docx file with page breaks" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(s"$docDirectory/page-breaks.docx")
    wordDf.select("doc").show(false)

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
    wordDf.select("doc").show(false)

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
    assert(!wordDf.columns.contains("content"))
  }

  "WordReader" should "read a docx file with images on it" taggedAs FastTest in {
    val wordReader = new WordReader()
    val wordDf = wordReader.doc(s"$docDirectory/contains-pictures.docx")
    wordDf.select("doc").show(false)

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
    assert(!wordDf.columns.contains("content"))
  }

  "WordReader" should "store content" taggedAs FastTest in {
    val wordReader = new WordReader(storeContent = true)
    val wordDf = wordReader.doc(s"$docDirectory")
    wordDf.select("doc").show(false)

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
    assert(wordDf.columns.contains("content"))
  }

}
