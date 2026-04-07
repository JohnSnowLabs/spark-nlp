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

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.reader.util.AssertReaders
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class RTFReaderTest extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  private val rtfDirectory = "src/test/resources/reader/rtf"
  private val sampleRtf = s"$rtfDirectory/sample.rtf"

  "RTFReader" should "read an rtf file with titles, paragraphs, and list items" taggedAs FastTest in {
    val rtfReader = new RTFReader()
    val rtfDf = rtfReader.rtf(sampleRtf)

    val elements = rtfDf
      .select(rtfReader.getOutputColumn)
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.exists(_.elementType == ElementType.TITLE))
    assert(elements.count(_.elementType == ElementType.LIST_ITEM) == 3)
    assert(elements.exists(_.content.contains("bold, italic, and underline")))
    assert(!rtfDf.columns.contains("content"))
  }

  it should "parse direct rtf content" taggedAs FastTest in {
    val source = Source.fromFile(sampleRtf, "UTF-8")
    val content =
      try source.mkString
      finally source.close()

    val elements = new RTFReader().rtfToHTMLElement(content)

    assert(elements.nonEmpty)
    assert(elements.head.elementType == ElementType.TITLE)
  }

  it should "store raw rtf content when configured" taggedAs FastTest in {
    val rtfReader = new RTFReader(storeContent = true)
    val rtfDf = rtfReader.rtf(sampleRtf)

    assert(rtfDf.columns.contains("content"))
  }

  it should "produce valid hierarchy metadata" taggedAs FastTest in {
    val rtfDf = new RTFReader().rtf(sampleRtf)
    AssertReaders.assertHierarchy(rtfDf, "rtf")
  }
}
