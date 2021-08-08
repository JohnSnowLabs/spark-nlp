/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence

import scala.collection.mutable.ArrayBuffer


/**
 * Works with different NER representations as tags
 * Supports: IOB and IOB2 https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
 */
object NerTagsEncoding {

  /**
   * Converts from IOB or IOB2 to list of NamedEntity
   * @param doc Source doc text
   * @return Extracted Named Entities
   */
  def fromIOB(sentence: NerTaggedSentence, doc: Annotation, sentenceIndex: Int = 0, originalOffset: Boolean = true,
              includeNoneEntities: Boolean = false): Seq[NamedEntity] = {
    val result = ArrayBuffer[NamedEntity]()

    val words = sentence.words.length

    var lastTag: Option[String] = None
    var lastTagStart = -1

    def flushEntity(startIdx: Int, endIdx: Int): Unit = {
      val start = sentence.indexedTaggedWords(startIdx).begin - doc.begin
      val end = sentence.indexedTaggedWords(endIdx).end - doc.begin
      require(start <= end && end <= doc.result.length, s"Failed to flush entities in NerConverter. " +
        s"Chunk offsets $start - $end are not within tokens:\n${sentence.words.mkString("||")}\nfor sentence:\n${doc.result}")
      val confidenceArray = sentence.indexedTaggedWords.slice(startIdx, endIdx + 1).flatMap(_.metadata.values)
      val finalConfidenceArray = try {
        confidenceArray.map(x => x.trim.toFloat)
      } catch {
        case _: Exception => Array.empty[Float]
      }
      val confidence = if(finalConfidenceArray.isEmpty) None else Some(finalConfidenceArray.sum / finalConfidenceArray.length)
      val content = if(originalOffset) doc.result.substring(start, end + 1) else sentence.indexedTaggedWords(startIdx).word
      val entity = NamedEntity(
        sentence.indexedTaggedWords(startIdx).begin,
        sentence.indexedTaggedWords(endIdx).end,
        lastTag.get,
        content,
        sentenceIndex.toString,
        confidence
      )
      result.append(entity)
      lastTag = None

    }

    def getTag(tag: String): Option[String] = {
      try {
        lastTag = Some(tag.substring(2))
      } catch {
        case e: StringIndexOutOfBoundsException =>
          require(tag.length < 2, s"This annotator only supports IOB and IOB2 tagging: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging) \n $e")
      }
      lastTag
    }

    for (i <- 0 until words) {
      val tag = sentence.tags(i)
      if (lastTag.isDefined && (tag.startsWith("B-") || tag == "O")) {
        flushEntity(lastTagStart, i - 1)
      }

      if (includeNoneEntities && lastTag.isEmpty) {
        lastTag = if (tag == "O" ) Some(tag) else getTag(tag)
        lastTagStart = i
      } else {
        if (lastTag.isEmpty && tag != "O") {
          lastTag = getTag(tag)
          lastTagStart = i
        }
      }
    }

    if (lastTag.isDefined) {
      flushEntity(lastTagStart, words - 1)
    }

    result.toList
  }

}

case class NamedEntity(start: Int, end: Int, entity: String, text: String, sentenceId: String, confidence: Option[Float])
