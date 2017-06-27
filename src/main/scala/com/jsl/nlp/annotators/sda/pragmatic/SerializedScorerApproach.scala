package com.jsl.nlp.annotators.sda.pragmatic

import com.jsl.nlp.annotators.param.SerializedAnnotatorComponent

/**
  * Created by saif on 24/06/17.
  */
case class SerializedScorerApproach(
                                   sentimentDict: Map[String, String]
                                   ) extends SerializedAnnotatorComponent[PragmaticScorer] {
  override def deserialize: PragmaticScorer = new PragmaticScorer(sentimentDict)
}