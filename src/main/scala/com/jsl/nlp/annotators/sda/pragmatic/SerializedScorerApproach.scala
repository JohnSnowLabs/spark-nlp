package com.jsl.nlp.annotators.sda.pragmatic

import com.jsl.nlp.annotators.param.SerializedAnnotatorApproach

/**
  * Created by saif on 24/06/17.
  */
case class SerializedScorerApproach(
                                   id: String,
                                   sentimentDict: Map[String, String]
                                   ) extends SerializedAnnotatorApproach[PragmaticScorer] {
  override def deserialize: PragmaticScorer = new PragmaticScorer(sentimentDict)
}
object SerializedScorerApproach {
  val id: String = "sentimentscorer"
}
