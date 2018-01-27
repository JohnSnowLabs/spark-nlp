package com.johnsnowlabs.nlp.annotators.assertion.logreg

/**
  * Created by jose on 18/12/17.
  */
class SimpleTokenizer extends Tokenizer {
  /* Tokenize a sentence splitting on spaces */
  def tokenize(sent: String) : Array[String] = sent.split(" ").map(_.trim).filter(_ != "")
}
