package com.jsl.nlp.annotators.sbd

import com.jsl.nlp.annotators.common.WritableAnnotatorComponent

/**
  * Created by Saif Addin on 5/5/2017.
  */

/**
  * Guideline interface for a sentence boundary approach
  */
trait SBDApproach extends WritableAnnotatorComponent {

  val description: String

  def extractBounds(text: String): Array[Sentence]

}