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

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.reader.{ElementType, HTMLElement}

import scala.collection.mutable

object TitleChunker {

  /** Splits a list of HTML elements into semantically grouped Chunks based on Title and Table
    * markers.
    *
    * @param elements
    *   List of input HTML elements to chunk.
    * @param maxCharacters
    *   Maximum length allowed per chunk. Longer sections are split.
    * @param combineTextUnderNChars
    *   Threshold to merge adjacent small sections.
    * @param overlap
    *   Number of characters to repeat between consecutive chunks.
    * @param newAfterNChars
    *   Soft limit to trigger new section if length exceeded, even before maxCharacters.
    * @param overlapAll
    *   Apply overlap context between all sections, not just split chunks.
    * @return
    *   List of Chunks partitioned by title and content heuristics.
    */
  def chunkByTitle(
      elements: List[HTMLElement],
      maxCharacters: Int,
      combineTextUnderNChars: Int = 0,
      overlap: Int = 0,
      newAfterNChars: Int = -1,
      overlapAll: Boolean = false): List[Chunk] = {

    val inputs = elements.map { element =>
      TitleChunkInput(
        source = element,
        text = element.content,
        metadata = element.metadata.toMap,
        elementType = Option(element.elementType),
        pageNumber =
          element.metadata.get("pageNumber").flatMap(v => scala.util.Try(v.toInt).toOption))
    }

    TitleChunkingUtil
      .chunk(
        inputs,
        TitleChunkingOptions(
          joinString = " ",
          splitOnPageChange = true,
          combineTextUnderNChars = combineTextUnderNChars,
          enableOverflowSplitting = true,
          maxCharacters = maxCharacters,
          newAfterNChars = newAfterNChars,
          overlap = overlap,
          overlapAll = overlapAll))
      .map { section =>
        val elements = section.items.map(_.source)
        if (elements.length == 1 &&
          elements.head.elementType.equalsIgnoreCase(ElementType.TABLE)) {
          Chunk(elements.toList)
        } else {
          val baseMetadata =
            if (elements.nonEmpty)
              mutable.Map.empty[String, String] ++ elements.head.metadata.toSeq
            else mutable.Map.empty[String, String]
          Chunk(List(HTMLElement(ElementType.NARRATIVE_TEXT, section.text, baseMetadata)))
        }
      }
      .toList
  }

  private[johnsnowlabs] def chunkAnnotationsByTitle(
      annotations: Seq[Annotation],
      joinString: String = " ",
      splitOnPageChange: Boolean = false,
      enableOverflowSplitting: Boolean = false,
      maxCharacters: Int = 500): Seq[TitleChunkSection[Annotation]] = {

    val inputs = annotations.map { annotation =>
      TitleChunkInput(
        source = annotation,
        text = annotation.result,
        metadata = annotation.metadata.toMap,
        elementType = annotation.metadata.get("elementType"),
        pageNumber =
          annotation.metadata.get("pageNumber").flatMap(v => scala.util.Try(v.toInt).toOption))
    }

    TitleChunkingUtil.chunk(
      inputs,
      TitleChunkingOptions(
        joinString = joinString,
        splitOnPageChange = splitOnPageChange,
        enableOverflowSplitting = enableOverflowSplitting,
        maxCharacters = maxCharacters))
  }

}
