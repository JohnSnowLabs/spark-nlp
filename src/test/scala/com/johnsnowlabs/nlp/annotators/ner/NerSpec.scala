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

import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence
import com.johnsnowlabs.nlp.annotators.common.IndexedTaggedWord
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.tags.FastTest

import org.scalatest.FlatSpec

import scala.collection.mutable.ArrayBuffer


class NerSpec extends FlatSpec {
  val doc = "word1 word2 word3 word4"

  private def createTagged(doc: String, tags: String*): NerTaggedSentence = {
    val words = doc.split(" ")
    val tokens = ArrayBuffer[IndexedTaggedWord]()

    var idx = 0
    for (i <- 0 until words.length) {
      val token = IndexedTaggedWord(words(i), tags(i), idx, idx + words(i).length - 1)
      tokens.append(token)

      idx = idx + words(i).length + 1
    }

    new NerTaggedSentence(tokens.toArray)
  }

  private def createEntities(doc: String, entities: (String, Int)*): Seq[NamedEntity] = {
    val words = doc.split(" ")

    var wordIdx = 0
    var idx = 0
    val result = ArrayBuffer[NamedEntity]()

    for ((entity, cnt) <- entities) {
      val start = idx

      for (i <- wordIdx until wordIdx + cnt) {
        idx += (words(i).length + 1)
      }

      wordIdx = wordIdx + cnt
      if (entity != "O") {
        val extracted = NamedEntity(start, idx - 2, entity, doc.substring(start, idx - 1), "0", None)
        result.append(extracted)
      }
    }

    result.toList
  }


  "NerTagsEncoder" should "correct Begin after Begin" taggedAs FastTest in {
    val tagged = createTagged(doc, "B-PER", "B-PER", "I-PER", "O")
    val parsed = NerTagsEncoding.fromIOB(tagged, Annotation(AnnotatorType.DOCUMENT, 0, doc.length - 1, doc, Map()))
    val target = createEntities(doc, ("PER", 1), ("PER", 2), ("O", 1))

    assert(parsed == target)
  }

  "NerTagsEncoder" should "correct process end of the sentence" taggedAs FastTest in {
    val tagged = createTagged(doc, "B-PER", "O", "B-PER", "I-PER")
    val parsed = NerTagsEncoding.fromIOB(tagged, Annotation(AnnotatorType.DOCUMENT, 0, doc.length - 1, doc, Map()))
    val target = createEntities(doc, ("PER", 1), ("O", 1), ("PER", 2))

    assert(parsed == target)
  }

}
