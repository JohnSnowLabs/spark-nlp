/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

case class DependencyParsedSentence(tokens: Array[WordWithDependency])

case class WordWithDependency(word: String, begin: Int, end: Int, head: Int)

object DependencyParsed extends Annotated[DependencyParsedSentence] {

  override def annotatorType: String = AnnotatorType.DEPENDENCY

  override def unpack(annotations: Seq[Annotation]): Seq[DependencyParsedSentence] = {
    val sentences = TokenizedWithSentence.unpack(annotations)
    val depAnnotations = annotations
      .filter(a => a.annotatorType == annotatorType)
      .sortBy(a => a.begin)

    var last = 0
    sentences.map { sentence =>
      val sorted = sentence.indexedTokens.sortBy(t => t.begin)
      val dependencies = (last until (last + sorted.length)).map { i =>
        depAnnotations(i).metadata("head").toInt
      }

      last += sorted.length

      val words = sorted.zip(dependencies).map { case (token, dependency) =>
        WordWithDependency(token.token, token.begin, token.end, dependency)
      }

      DependencyParsedSentence(words)
    }
  }

  override def pack(items: Seq[DependencyParsedSentence]): Seq[Annotation] = {
    items.zipWithIndex.flatMap { case (sentence, index) =>
      sentence.tokens.map { token =>
        val headData = getHeadData(token.head, sentence)
        val realHead = if (token.head == sentence.tokens.length) 0 else token.head + 1
        Annotation(
          annotatorType,
          token.begin,
          token.end,
          headData.word,
          Map(
            "head" -> realHead.toString,
            "head.begin" -> headData.begin.toString,
            "head.end" -> headData.end.toString,
            "sentence" -> index.toString))
      }
    }
  }

  def getHeadData(head: Int, sentence: DependencyParsedSentence): WordWithDependency = {
    val root: WordWithDependency = WordWithDependency("ROOT", -1, -1, -1)
    sentence.tokens.lift(head).getOrElse(root)
  }

}
