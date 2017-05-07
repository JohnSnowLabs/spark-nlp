package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.annotators.sbd.{SBDApproach, Sentence}

/**
  * Created by Saif Addin on 5/5/2017.
  */
class PragmaticDetection(target: String) extends SBDApproach {

  override val description: String = "rule-based sentence detector. based off pragmatic sentence detector."

  private var wip: String = target

  override def prepare: SBDApproach = {
    wip = new PragmaticContentCleaner(target)
      .formatLists
      .formatNumbers
      .formatPunctuations
      .formatMultiplePeriods
      .formatGeoLocations
      .finish
    this
  }

  override def extract: Array[Sentence] = {
    new PragmaticSentenceExtractor(wip)
      .pull
  }

}



