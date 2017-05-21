package com.jsl.nlp.annotators.pos

/**
  * Created by Saif Addin on 5/20/2017.
  */
abstract class POSModel {

  def predict(features: Map[String, Int]): String

}
