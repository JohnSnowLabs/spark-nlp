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
import org.apache.spark.sql.functions.col
import org.scalatest.flatspec.AnyFlatSpec

class HTMLReaderTest extends AnyFlatSpec {

  val htmlFilesDirectory = "./src/test/resources/reader/html/"

  it should "read html as dataframe" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val result = HTMLReader.read(htmlFilesDirectory)
    result.show()
  }

  it should "read html as dataframe with params" taggedAs FastTest in {
    val HTMLReader = new HTMLReader(titleFontSize = 12)
    val htmlDF = HTMLReader.read(htmlFilesDirectory)
    htmlDF.show()

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
  }

  it should "parse an html in real time" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read("https://www.wikipedia.org")
    htmlDF.show()

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
  }

  it should "parse URLS in real time" taggedAs FastTest in {
    val HTMLReader = new HTMLReader()
    val htmlDF = HTMLReader.read(Array("https://www.wikipedia.org", "https://example.com/"))
    htmlDF.show()

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
  }

}
