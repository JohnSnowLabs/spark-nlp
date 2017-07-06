package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.annotators.param.SerializedAnnotatorComponent
import com.jsl.nlp.annotators.sbd.{SBDApproach, Sentence}

/**
  * Created by Saif Addin on 5/5/2017.
  */

/**
  * Inspired on Kevin Dias, Ruby implementation: https://github.com/diasks2/pragmatic_segmenter
  * This approach extracts sentence bounds by first formatting the data with [[RuleSymbols]] and then extracting bounds
  * with a strong RegexBased rule application
  */
class PragmaticApproach(useAbbreviations: Boolean = true) extends SBDApproach {

  override val description: String = "rule-based sentence detector. based off pragmatic sentence detector."

  override def extractBounds(content: String): Array[Sentence] = {
    /** this is a hardcoded order of operations
      * considered to go from those most specific non-ambiguous cases
      * down to those that are more general and can easily be ambiguous
      */
    val symbolyzedData = new PragmaticContentFormatter(content)
      .formatLists
      .formatAbbreviations(useAbbreviations)
      .formatNumbers
      .formatPunctuations
      .formatMultiplePeriods
      .formatGeoLocations
      .formatEllipsisRules
      .formatBetweenPunctuations
      .formatDoublePunctuations
      .formatQuotationMarkInQuotation
      .formatExclamationPoint
      .formatBasicBreakers
      .finish
    new PragmaticSentenceExtractor(symbolyzedData).pull
  }

  /** Serializable representation of this model. Doesn't really have any special parts */
  override def serialize: SerializedAnnotatorComponent[PragmaticApproach] = SerializedSBDApproach()

}
