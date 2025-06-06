/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.partition

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.explode
import org.scalatest.flatspec.AnyFlatSpec

class PartitionChunkerTest extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._
  val txtDirectory = "src/test/resources/reader/txt"
  val htmlDirectory = "src/test/resources/reader/html"

  "Partition" should "perform basic chunk text" taggedAs FastTest in {
    val partitionOptions = Map("contentType" -> "text/plain", "chunkingStrategy" -> "basic")
    val textDf = Partition(partitionOptions).partition(s"$txtDirectory/long-text.txt")
    textDf.show(truncate = false)

    val partitionDf = textDf.select(explode($"txt.content"))
    partitionDf.show(truncate = false)
    assert(partitionDf.count() == 1)

    val chunkDf = textDf.select(explode($"chunks.content"))
    chunkDf.show(truncate = false)
    assert(chunkDf.count() > 1)
  }

  it should "perform chunking by title" taggedAs FastTest in {
    val partitionOptions = Map(
      "contentType" -> "text/html",
      "titleFontSize" -> "14",
      "chunkingStrategy" -> "byTitle",
      "combineTextUnderNChars" -> "50")
    val textDf = Partition(partitionOptions).partition(s"$htmlDirectory/fake-html.html")

    val partitionDf = textDf.select(explode($"chunks.content"))
    partitionDf.show(truncate = false)
    assert(partitionDf.count() == 2)
  }

}
