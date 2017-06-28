package com.jsl.nlp.annotators.pos

/**
  * Created by Saif Addin on 5/20/2017.
  */

/**
  * Representation for a POS model. For now, having a predict operation is the only needed interface
  * Features required for prediction are of any type
  */
abstract class POSModel[A] {

  def predict(features: A): String

}
