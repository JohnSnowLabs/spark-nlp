package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.annotators.param.SerializedAnnotatorApproach
import com.jsl.nlp.annotators.sbd.{SBDApproach, Sentence}

/**
  * Created by Saif Addin on 5/5/2017.
  * Inspired on Kevin Dias, Ruby implementation: https://github.com/diasks2/pragmatic_segmenter
  */
class PragmaticApproach extends SBDApproach {

  override val description: String = "rule-based sentence detector. based off pragmatic sentence detector."

  override def prepare: SBDApproach = {
    updateContent(
      new PragmaticContentFormatter(getContent.getOrElse(
        throw new NoSuchElementException("setInput not called before prepare"))
      )
        .formatLists
        .formatAbbreviations
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
    )
    this
  }

  override def serialize: SerializedAnnotatorApproach[PragmaticApproach] =
    SerializedSBDApproach(SerializedSBDApproach.id)

  override def extract: Array[Sentence] = {
    new PragmaticSentenceExtractor(getContent
      .getOrElse(throw new NoSuchElementException("setInput not called before prepare"))
    ).pull
  }

}
