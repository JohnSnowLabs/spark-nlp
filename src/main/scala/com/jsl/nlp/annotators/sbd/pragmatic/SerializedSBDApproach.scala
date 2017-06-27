package com.jsl.nlp.annotators.sbd.pragmatic

import com.jsl.nlp.annotators.param.SerializedAnnotatorComponent

/**
  * Created by saif on 24/06/17.
  */

/**
  * Serialized representation of [[PragmaticApproach]]. Nothing special here.
  */
case class SerializedSBDApproach() extends SerializedAnnotatorComponent[PragmaticApproach] {
  override def deserialize: PragmaticApproach = new PragmaticApproach
}