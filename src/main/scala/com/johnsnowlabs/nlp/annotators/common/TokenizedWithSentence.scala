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

package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}


object TokenizedWithSentence extends Annotated[TokenizedSentence] {

  override def annotatorType: String = AnnotatorType.TOKEN

  override def unpack(annotations: Seq[Annotation]): Seq[TokenizedSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .toArray

    val sentences = SentenceSplit.unpack(annotations)

    /** // Evaluate whether to enable this validation to check proper usage of DOCUMENT and SENTENCE within entire pipelines
    require(tokens.map(_.metadata.getOrElse("sentence", "0").toInt).distinct.length == sentences.length,
      "Inconsistencies found in pipeline. Tokens in sentences does not match with sentence count")
      */

    sentences.map(sentence => {
      val sentenceTokens = tokens.filter(token =>
        token.begin >= sentence.start & token.end <= sentence.end
      ).map(token => IndexedToken(token.result, token.begin, token.end))
      sentenceTokens
    }).zipWithIndex.map{case (indexedTokens, index) => TokenizedSentence(indexedTokens, index)}.filter(_.indexedTokens.nonEmpty)
  }

  override def pack(sentences: Seq[TokenizedSentence]): Seq[Annotation] = {
    sentences.flatMap{ sentence =>
        sentence.indexedTokens.map{token =>
        Annotation(annotatorType, token.begin, token.end, token.token,
          Map("sentence" -> sentence.sentenceIndex.toString))
    }}
  }
}
