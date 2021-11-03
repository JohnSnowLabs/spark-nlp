/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition

import com.johnsnowlabs.nlp.annotators.common.{DependencyParsedSentence, WordWithDependency}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence

object GreedyTransitionApproach {

  def predict(posTagged: PosTaggedSentence, dependencyMaker: DependencyMaker): DependencyParsedSentence = {
    val sentence: Sentence = posTagged.indexedTaggedWords
      .map { item => WordData(item.word, item.tag) }.toList
    val dependencies = dependencyMaker.parse(sentence)
    val words = posTagged.indexedTaggedWords
      .zip(dependencies)
      .map{
        case (word, dependency) =>
          WordWithDependency(word.word, word.begin, word.end, dependency)
      }

    DependencyParsedSentence(words)
  }

}
