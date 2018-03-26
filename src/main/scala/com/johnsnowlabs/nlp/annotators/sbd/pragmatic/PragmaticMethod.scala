package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.annotators.common.Sentence

/**
  * Created by Saif Addin on 5/5/2017.
  */

/**
  * Inspired on Kevin Dias, Ruby implementation: https://github.com/diasks2/pragmatic_segmenter
  * This approach extracts sentence bounds by first formatting the data with [[RuleSymbols]] and then extracting bounds
  * with a strong RegexBased rule application
  */
class PragmaticMethod(useAbbreviations: Boolean = false) extends Serializable {

  def extractBounds(content: String, customBounds: Array[String]): Array[Sentence] = {
    /** this is a hardcoded order of operations
      * considered to go from those most specific non-ambiguous cases
      * down to those that are more general and can easily be ambiguous
      */
    val symbolyzedData = new PragmaticContentFormatter(content)
      .formatCustomBounds(customBounds)
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
