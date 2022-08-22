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

package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.util.regex.MatchStrategy.MATCH_ALL
import com.johnsnowlabs.nlp.util.regex.RuleFactory
import com.johnsnowlabs.nlp.util.regex.TransformStrategy.{
  REPLACE_ALL_WITH_SYMBOL,
  TransformStrategy
}

protected trait PragmaticMethod {
  def extractBounds(content: String): Array[Sentence]
}

/** Inspired on Kevin Dias, Ruby implementation: https://github.com/diasks2/pragmatic_segmenter
  * This approach extracts sentence bounds by first formatting the data with [[RuleSymbols]] and
  * then extracting bounds with a strong RegexBased rule application
  */
class CustomPragmaticMethod(
    customBounds: Array[String],
    transformStrategy: TransformStrategy = REPLACE_ALL_WITH_SYMBOL)
    extends PragmaticMethod
    with Serializable {
  override def extractBounds(content: String): Array[Sentence] = {

    val customBoundsFactory = new RuleFactory(MATCH_ALL, transformStrategy)
    customBounds.foreach(bound => customBoundsFactory.addRule(bound.r, s"split bound: $bound"))

    val symbolyzedData = new PragmaticContentFormatter(content)
      .formatCustomBounds(customBoundsFactory)
      .finish

    new PragmaticSentenceExtractor(symbolyzedData, content).pull
  }
}

class DefaultPragmaticMethod(useAbbreviations: Boolean = false, detectLists: Boolean = true)
    extends PragmaticMethod
    with Serializable {

  /** this is a hardcoded order of operations considered to go from those most specific
    * non-ambiguous cases down to those that are more general and can easily be ambiguous
    */
  def extractBounds(content: String): Array[Sentence] = {
    val symbolyzedData =
      new PragmaticContentFormatter(content)
        .formatLists(detectLists)
        .formatNumbers
        .formatAbbreviations(useAbbreviations)
        .formatPunctuations
        .formatMultiplePeriods
        .formatGeoLocations
        .formatEllipsisRules
        .formatBetweenPunctuations
        .formatQuotationMarkInQuotation
        .formatExclamationPoint
        .formatBasicBreakers
        .finish
    new PragmaticSentenceExtractor(symbolyzedData, content).pull
  }
}

class MixedPragmaticMethod(
    useAbbreviations: Boolean = false,
    detectLists: Boolean = true,
    customBounds: Array[String],
    transformStrategy: TransformStrategy = REPLACE_ALL_WITH_SYMBOL)
    extends PragmaticMethod
    with Serializable {
  val customBoundsFactory = new RuleFactory(MATCH_ALL, transformStrategy)
  customBounds.foreach(bound => customBoundsFactory.addRule(bound.r, s"split bound: $bound"))

  /** this is a hardcoded order of operations considered to go from those most specific
    * non-ambiguous cases down to those that are more general and can easily be ambiguous
    */
  def extractBounds(content: String): Array[Sentence] = {
    val symbolyzedData =
      new PragmaticContentFormatter(content)
        .formatCustomBounds(customBoundsFactory)
        .formatLists(detectLists)
        .formatAbbreviations(useAbbreviations)
        .formatNumbers
        .formatPunctuations
        .formatMultiplePeriods
        .formatGeoLocations
        .formatEllipsisRules
        .formatBetweenPunctuations
        .formatQuotationMarkInQuotation
        .formatExclamationPoint
        .formatBasicBreakers
        .finish
    new PragmaticSentenceExtractor(symbolyzedData, content).pull
  }
}
