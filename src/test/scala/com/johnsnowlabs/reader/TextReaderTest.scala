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
package com.johnsnowlabs.reader

import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.col
import org.scalatest.flatspec.AnyFlatSpec

class TextReaderTest extends AnyFlatSpec {

  val txtDirectory = "src/test/resources/reader/txt/"

  "Text Reader" should "read a directory of text files" taggedAs FastTest in {
    val textReader = new TextReader()
    val textDf = textReader.txt(s"$txtDirectory/simple-text.txt")
    textDf.select("txt").show(false)

    assert(!textDf.select(col("txt").getItem(0)).isEmpty)
    assert(!textDf.columns.contains("content"))
  }

  "Text Reader" should "store content" taggedAs FastTest in {
    val textReader = new TextReader(storeContent = true)
    val textDf = textReader.txt(txtDirectory)
    textDf.show()

    assert(!textDf.select(col("txt").getItem(0)).isEmpty)
    assert(textDf.columns.contains("content"))
  }

}
