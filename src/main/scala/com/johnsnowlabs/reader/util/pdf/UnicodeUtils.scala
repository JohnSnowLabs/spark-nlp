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
package com.johnsnowlabs.reader.util.pdf

import com.johnsnowlabs.reader.util.pdf.schema.MappingMatrix

class UnicodeUtils {
  // supported ligatures and their mappings(replacements) here
  val ligatures = Map(
    'ﬀ' -> Array("f", "f"),
    'ﬁ' -> Array("f", "i"),
    'ﬂ' -> Array("f", "l"),
    'ﬃ' -> Array("f", "f", "i"),
    'ﬄ' -> Array("f", "f", "l"),
    'ﬆ' -> Array("s", "t"),
    'œ' -> Array("o", "e"))

  /*
     Replace the following ligatures:
     {"fi", "fl", "ff", "ffi", "ffl", "st", "ft", "oe", "OE"},
     with their corresponding characters.
   */

  def splitOnLigature(mapping: MappingMatrix): Array[MappingMatrix] = {
    // split on the first ligature
    val firstLigatureIdx = mapping.c.indexWhere(char => ligatures.contains(char))

    firstLigatureIdx match {
      case -1 => Array(mapping)

      // handling the case for ligatures of 2 and 3 chars differently.
      case _ if ligatures(mapping.c(firstLigatureIdx)).length == 2 =>
        val firstChunk = mapping.c.substring(0, firstLigatureIdx)
        val secondChunk = mapping.c.substring(firstLigatureIdx + 1, mapping.c.length)
        val approxWidthFirstChunk = (mapping.width / mapping.c.length) * firstChunk.length
        val Array(firstChar, secondChar) = ligatures(mapping.c(firstLigatureIdx))

        val firstMapping = MappingMatrix(
          firstChunk + firstChar,
          mapping.x,
          mapping.y,
          approxWidthFirstChunk,
          mapping.height,
          mapping.fontSize,
          mapping.source)

        val secondMapping = MappingMatrix(
          secondChar + secondChunk,
          mapping.x + approxWidthFirstChunk,
          mapping.y,
          mapping.width - approxWidthFirstChunk,
          mapping.height,
          mapping.fontSize,
          mapping.source)

        // in case chunks have in turn more ligatures on each chunk
        Array(firstMapping, secondMapping).flatMap(splitOnLigature)

      case _ if ligatures(mapping.c(firstLigatureIdx)).length == 3 =>
        // this are not 'Char's
        val Array(firstChar, secondChar, thirdChar) = ligatures(mapping.c(firstLigatureIdx))

        val firstChunk = mapping.c.substring(0, firstLigatureIdx)
        val thirdChunk = mapping.c.substring(firstLigatureIdx + 1, mapping.c.length)
        val approxWidthFirstChunk = (mapping.width / mapping.c.length) * firstChunk.length

        // single char, length 1
        val approxWidthSecondChunk = (mapping.width / mapping.c.length)

        val firstMapping = MappingMatrix(
          firstChunk + firstChar,
          mapping.x,
          mapping.y,
          approxWidthFirstChunk,
          mapping.height,
          mapping.fontSize,
          mapping.source)

        val secondMapping = MappingMatrix(
          secondChar,
          mapping.x + approxWidthFirstChunk + approxWidthSecondChunk / 3,
          mapping.y,
          approxWidthSecondChunk / 3,
          mapping.height,
          mapping.fontSize,
          mapping.source)

        val thirdMapping = MappingMatrix(
          thirdChar + thirdChunk,
          mapping.x + approxWidthFirstChunk +
            approxWidthSecondChunk,
          mapping.y,
          mapping.width - approxWidthFirstChunk - approxWidthSecondChunk,
          mapping.height,
          mapping.fontSize,
          mapping.source)

        // in case chunks have in turn more ligatures on each chunk
        Array(firstMapping, secondMapping, thirdMapping).flatMap(splitOnLigature)

    }
  }

  def normalizeLigatures(input: Array[MappingMatrix]) =
    input.flatMap { mapping =>
      splitOnLigature(mapping)
    }
}
