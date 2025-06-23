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
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class MarkdownReaderTest extends AnyFlatSpec {

  val mdDirectory = "src/test/resources/reader/md"

  "Markdown Reader" should "read a markdown file with headers and text" taggedAs FastTest in {
    import ResourceHelper.spark.implicits._
    val reader = new MarkdownReader()
    val textDf = reader.md(s"$mdDirectory/simple.md")
    textDf.select("md").show(truncate = false)

    val elements: Seq[HTMLElement] = textDf
      .select("md")
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.exists(_.elementType == ElementType.TITLE))
    assert(elements.exists(_.elementType == ElementType.NARRATIVE_TEXT))
  }

  it should "detect list items in markdown" taggedAs FastTest in {
    val reader = new MarkdownReader()
    val content = """
                    |# Shopping List
                    | - Milk
                    | - Bread
                    | - Eggs
                    |""".stripMargin

    val items = reader.parseMarkdown(content)

    assert(items.count(_.elementType == ElementType.LIST_ITEM) == 3)
    assert(items.exists(e => e.content.contains("Bread")))
  }

  it should "parse code blocks in markdown" taggedAs FastTest in {
    val reader = new MarkdownReader()
    val content = """
                    |```scala
                    |val x = 10
                    |println(x)
                    |```
                    |""".stripMargin

    val items = reader.parseMarkdown(content)

    assert(items.exists(_.elementType == ElementType.UNCATEGORIZED_TEXT))
    assert(items.exists(_.content.contains("val x = 10")))
  }

  it should "assign paragraph metadata correctly" taggedAs FastTest in {
    val reader = new MarkdownReader()
    val content = """
                    |# Intro
                    |Some intro text.
                    |
                    |## Details
                    |Detail line one.
                    |Detail line two.
                    |""".stripMargin

    val elements = reader.parseMarkdown(content)
    val paragraphs = elements.map(_.metadata("paragraph")).toSet
    assert(paragraphs.size > 1)
  }

}
