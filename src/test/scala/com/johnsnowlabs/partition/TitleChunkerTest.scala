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
package com.johnsnowlabs.partition

import com.johnsnowlabs.reader.HTMLElement
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class TitleChunkerTest extends AnyFlatSpec {

  def element(et: String, text: String, page: Int = 1): HTMLElement =
    HTMLElement(et, text, mutable.Map("pageNumber" -> page.toString))

  "chunkByTitle" should "start a new chunk when a title element is found" in {
    val elements = List(
      element("Title", "My First Heading"),
      element("NarrativeText", "My first paragraph. lorem ipsum dolor set amet."),
      element("Title", "My Second Heading"),
      element("NarrativeText", "My second paragraph."))

    val result = TitleChunker.chunkByTitle(elements, maxCharacters = 1000)

    assert(result.length == 2)
    assert(result.head.elements.head.content.contains("My First Heading"))
    assert(result.head.elements.head.content.contains("My first paragraph"))
    assert(result(1).elements.head.content.contains("My Second Heading"))
    assert(result(1).elements.head.content.contains("My second paragraph"))
  }

  it should "split on soft limit newAfterNChars" in {
    val elements = List(
      element("Title", "Heading"),
      element("NarrativeText", "a " * 50),
      element("NarrativeText", "b " * 50))

    val result = TitleChunker.chunkByTitle(elements, maxCharacters = 1000, newAfterNChars = 100)

    assert(result.length == 2)
  }

  it should "add overlap context when overlapAll is true" in {
    val elements = List(
      element("Title", "Intro"),
      element("NarrativeText", "The cow jumped over the moon. " * 5),
      element("Title", "Next Section"),
      element("NarrativeText", "And the dish ran away with the spoon."))

    val maxCharacters = 100
    val overlap = 10
    val result = TitleChunker.chunkByTitle(
      elements,
      maxCharacters = maxCharacters,
      overlap = overlap,
      overlapAll = true)
    assert(result.length >= 2)

    val prevText = ("The cow jumped over the moon. " * 5).trim
    val expectedOverlap = prevText.takeRight(overlap).trim
    assert(result(1).elements.head.content.contains(expectedOverlap))
  }

  it should "chunk content correctly across page boundaries" in {
    val elements = List(
      element("Title", "Page 1 Heading"),
      element("NarrativeText", "Text on page 1."),
      element("Title", "Page 2 Heading", page = 2),
      element("NarrativeText", "Text on page 2.", page = 2))

    val result = TitleChunker.chunkByTitle(elements, maxCharacters = 1000)
    assert(result.length == 2)
    assert(result(0).elements.head.content.contains("Page 1 Heading"))
    assert(result(1).elements.head.content.contains("Page 2 Heading"))
  }

}
