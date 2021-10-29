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

package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, IndexedToken, TaggedSentence, TokenizedSentence}

trait PerceptronPredictionUtils extends PerceptronUtils {

  /**
   * Tags a group of sentences into POS tagged sentences
   * The logic here is to create a sentence context, run through every word and evaluate its context
   * Based on how frequent a context appears around a word, such context is given a score which is used to predict
   * Some words are marked as non ambiguous from the beginning
   *
   * @param tokenizedSentences Sentence in the form of single word tokens
   * @return A list of sentences which have every word tagged
   */
  def tag(model: AveragedPerceptron, tokenizedSentences: Array[TokenizedSentence]): Array[TaggedSentence] = {
    //logger.debug(s"PREDICTION: Tagging:\nSENT: <<${tokenizedSentences.map(_.condense).mkString(">>\nSENT<<")}>> model weight properties in 'bias' " +
    //s"feature:\nPREDICTION: ${$$(model).getWeights("bias").mkString("\nPREDICTION: ")}")
    var prev = START(0)
    var prev2 = START(1)
    tokenizedSentences.map(sentence => {
      val context: Array[String] = START ++: sentence.tokens.map(normalized) ++: END
      sentence.indexedTokens.zipWithIndex.map { case (IndexedToken(word, begin, end), i) =>
        val tag = model.getTaggedBook.getOrElse(word.toLowerCase,
          {
            val features = getFeatures(i, word, context, prev, prev2)
            model.predict(features)
          }
        )
        prev2 = prev
        prev = tag
        IndexedTaggedWord(word, tag, begin, end, None, Map("index" -> i.toString))
      }
    }).map(TaggedSentence(_))
  }

}
