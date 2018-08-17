package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.util.regex.MatchStrategy.MATCH_ALL
import com.johnsnowlabs.nlp.util.regex.RuleFactory
import com.johnsnowlabs.nlp.util.regex.TransformStrategy.REPLACE_ALL_WITH_SYMBOL

/**
  * Created by Saif Addin on 5/5/2017.
  */

protected trait PragmaticMethod {
  def extractBounds(content: String): Array[Sentence]
}

/**
  * Inspired on Kevin Dias, Ruby implementation: https://github.com/diasks2/pragmatic_segmenter
  * This approach extracts sentence bounds by first formatting the data with [[RuleSymbols]] and then extracting bounds
  * with a strong RegexBased rule application
  */
class CustomPragmaticMethod(customBounds: Array[String]) extends PragmaticMethod with Serializable {
  override def extractBounds(content: String): Array[Sentence] = {

    val customBoundsFactory = new RuleFactory(MATCH_ALL, REPLACE_ALL_WITH_SYMBOL)
    customBounds.foreach(bound => customBoundsFactory.addRule(bound.r, s"split bound: $bound"))

    val symbolyzedData = new PragmaticContentFormatter(content)
        .formatCustomBounds(customBoundsFactory)
        .finish

    new PragmaticSentenceExtractor(symbolyzedData, content).pull
  }
}

class DefaultPragmaticMethod(useAbbreviations: Boolean = false) extends PragmaticMethod with Serializable {
  /** this is a hardcoded order of operations
    * considered to go from those most specific non-ambiguous cases
    * down to those that are more general and can easily be ambiguous
    */
  def extractBounds(content: String): Array[Sentence] = {
    val symbolyzedData =
      new PragmaticContentFormatter(content)
        .formatLists
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

class MixedPragmaticMethod(useAbbreviations: Boolean = false, customBounds: Array[String]) extends PragmaticMethod with Serializable {
  val customBoundsFactory = new RuleFactory(MATCH_ALL, REPLACE_ALL_WITH_SYMBOL)
  customBounds.foreach(bound => customBoundsFactory.addRule(bound.r, s"split bound: $bound"))
  /** this is a hardcoded order of operations
    * considered to go from those most specific non-ambiguous cases
    * down to those that are more general and can easily be ambiguous
    */
  def extractBounds(content: String): Array[Sentence] = {
    val symbolyzedData =
      new PragmaticContentFormatter(content)
        .formatCustomBounds(customBoundsFactory)
        .formatLists
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