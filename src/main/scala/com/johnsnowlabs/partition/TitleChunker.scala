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

import com.johnsnowlabs.reader.{ElementType, HTMLElement}

import scala.collection.mutable

object TitleChunker {

  /**
   * Splits a list of HTML elements into semantically grouped Chunks based on Title and Table markers.
   *
   * @param elements List of input HTML elements to chunk.
   * @param maxCharacters Maximum length allowed per chunk. Longer sections are split.
   * @param combineTextUnderNChars Threshold to merge adjacent small sections (if non-table).
   * @param overlap Number of characters to repeat between consecutive chunks.
   * @param newAfterNChars Soft limit to trigger new section if length exceeded, even before maxCharacters.
   * @param overlapAll Apply overlap context between all sections, not just split chunks.
   * @return List of Chunks partitioned by title and content heuristics.
   */
  def chunkByTitle(
                    elements: List[HTMLElement],
                    maxCharacters: Int,
                    combineTextUnderNChars: Int = 0,
                    overlap: Int = 0,
                    newAfterNChars: Int = -1,
                    overlapAll: Boolean = false): List[Chunk] = {

    val softLimit = if (newAfterNChars <= 0) maxCharacters else newAfterNChars
    val chunks = mutable.ListBuffer.empty[Chunk]
    val sections = mutable.ListBuffer.empty[List[HTMLElement]]
    var currentSection = List.empty[HTMLElement]
    var currentLength = 0
    var currentPage = -1

    for (element <- elements) {
      val elementLength = element.content.length
      val isTable = element.elementType == "Table"
      val elementPage = element.metadata.getOrElse("pageNumber", "-1").toInt

      val pageChanged = currentPage != -1 && elementPage != currentPage
      val softLimitExceeded = currentSection.length >= 2 &&
        (currentLength + elementLength > softLimit)

      if (isTable) {
        if (currentSection.nonEmpty) sections += currentSection
        sections += List(element)
        currentSection = List.empty
        currentLength = 0
        currentPage = -1
      } else if (pageChanged || softLimitExceeded) {
        if (currentSection.nonEmpty) sections += currentSection
        currentSection = List(element)
        currentLength = elementLength
        currentPage = elementPage
      } else {
        currentSection :+= element
        currentLength += elementLength
        currentPage = elementPage
      }
    }
    if (currentSection.nonEmpty) sections += currentSection

    val mergedSections = sections.foldLeft(List.empty[List[HTMLElement]]) { (acc, section) =>
      val sectionLength = section.map(_.content.length).sum
      val canMerge = combineTextUnderNChars > 0 &&
        sectionLength < combineTextUnderNChars &&
        acc.nonEmpty &&
        acc.last.exists(_.elementType != "Table") &&
        section.exists(_.elementType != "Table")

      if (canMerge) {
        acc.init :+ (acc.last ++ section)
      } else {
        acc :+ section
      }
    }

    var lastNarrativeText = ""
    for (section <- mergedSections) {
      if (section.exists(_.elementType == "Table")) {
        chunks += Chunk(section)
        lastNarrativeText = ""
      } else {
        val sectionText = section.map(_.content).mkString(" ")
        val content =
          if (overlap > 0 && lastNarrativeText.nonEmpty && (overlapAll || sectionText.length > maxCharacters))
            lastNarrativeText.takeRight(overlap) + " " + sectionText
          else sectionText

        val merged = HTMLElement(ElementType.NARRATIVE_TEXT, content.trim, section.head.metadata)
        val split = if (content.length > maxCharacters) {
          splitHTMLElement(merged, maxCharacters, overlap)
        } else List(merged)

        chunks ++= split.map(e => Chunk(List(e)))
        lastNarrativeText = sectionText
      }
    }

    chunks.toList
  }

  private def splitHTMLElement(
                                element: HTMLElement,
                                maxLen: Int,
                                overlap: Int): List[HTMLElement] = {

    val words = element.content.split(" ")
    val buffer = mutable.ListBuffer.empty[HTMLElement]
    var chunk = new StringBuilder

    for (word <- words) {
      if (chunk.length + word.length + 1 > maxLen) {
        val text = chunk.toString().trim
        buffer += element.copy(content = text)
        chunk = new StringBuilder
        if (overlap > 0 && text.length >= overlap)
          chunk.append(text.takeRight(overlap)).append(" ")
      }
      chunk.append(word).append(" ")
    }

    if (chunk.nonEmpty)
      buffer += element.copy(content = chunk.toString().trim)

    buffer.toList
  }

}
