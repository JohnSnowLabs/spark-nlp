package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.annotators.sbd.pragmatic.rule.PragmaticContentFormatter
import com.jsl.nlp.annotators.sbd.{SBDApproach, Sentence}

/**
  * Created by Saif Addin on 5/5/2017.
  * Inspired on Kevin Dias, Ruby implementation: https://github.com/diasks2/pragmatic_segmenter
  */
class PragmaticDetection(target: String) extends SBDApproach {

  override val description: String = "rule-based sentence detector. based off pragmatic sentence detector."

  private var wip: String = target

  override def prepare: SBDApproach = {
    wip = new PragmaticContentFormatter(target)
      .formatLists
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
    this
  }

  override def extract: Array[Sentence] = {
    new PragmaticSentenceExtractor(wip)
      .pull
  }

}



