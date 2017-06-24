package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.annotators.param.SerializedAnnotatorApproach

/**
  * Created by saif on 24/06/17.
  */
case class SerializedSBDApproach(
                                id: String
                                ) extends SerializedAnnotatorApproach[PragmaticApproach] {
  override def deserialize: PragmaticApproach = new PragmaticApproach
}
object SerializedSBDApproach {
  val id: String = "pragmaticsbd"
}