package com.johnsnowlabs.nlp.annotators.assertion.logreg

/**
  * Created by jose on 18/12/17.
  */
trait Tokenizer {

  def tokenize(sent: String) : Array[String]



}
