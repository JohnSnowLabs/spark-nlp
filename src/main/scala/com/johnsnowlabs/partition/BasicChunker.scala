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

import com.johnsnowlabs.reader.HTMLElement

import scala.collection.mutable

case class Chunk(elements: List[HTMLElement]) {
  def length: Int = elements.map(_.content.length).sum
}

object BasicChunker {

  /** Splits a list of [[HTMLElement]]s into chunks constrained by a maximum number of characters.
    *
    * This method ensures that no chunk exceeds the specified `maxCharacters` limit. Optionally, a
    * `newAfterNChars` parameter can be used to set a soft boundary for starting new chunks
    * earlier, and `overlap` can be used to retain trailing characters from the previous chunk in
    * the next one (when splitting long elements).
    *
    * @param elements
    *   The list of [[HTMLElement]]s to be chunked.
    * @param maxCharacters
    *   The hard limit on the number of characters per chunk.
    * @param newAfterNChars
    *   Optional soft limit for starting a new chunk before reaching `maxCharacters`. If set to
    * -1, this soft limit is ignored.
    * @param overlap
    *   Number of trailing characters to overlap between chunks when splitting long elements. This
    *   helps maintain context in downstream NLP tasks.
    * @return
    *   A list of [[Chunk]] objects, each containing a group of elements whose combined content
    *   length does not exceed the specified limits.
    */

  def chunkBasic(
      elements: List[HTMLElement],
      maxCharacters: Int,
      newAfterNChars: Int = -1,
      overlap: Int = 0): List[Chunk] = {
    val softLimit = if (newAfterNChars > 0) newAfterNChars else maxCharacters
    var currentChunk = List.empty[HTMLElement]
    var currentLength = 0
    val chunks = mutable.ListBuffer.empty[Chunk]

    def finalizeChunk(): Unit = {
      if (currentChunk.nonEmpty) {
        chunks += Chunk(currentChunk)
        currentChunk = List.empty[HTMLElement]
        currentLength = 0
      }
    }

    for (element <- elements) {
      val elLength = element.content.length

      if (elLength > maxCharacters) {
        val splitElements = splitHTMLElement(element, maxCharacters, overlap)
        for (splitEl <- splitElements) {
          if (currentLength + splitEl.content.length > maxCharacters || currentLength >= softLimit)
            finalizeChunk()
          currentChunk :+= splitEl
          currentLength += splitEl.content.length
        }
      } else if (currentLength + elLength > maxCharacters || currentLength >= softLimit) {
        finalizeChunk()
        currentChunk :+= element
        currentLength += elLength
      } else {
        currentChunk :+= element
        currentLength += elLength
      }
    }

    finalizeChunk()
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
