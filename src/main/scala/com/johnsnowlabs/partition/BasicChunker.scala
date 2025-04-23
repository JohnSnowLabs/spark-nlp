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
