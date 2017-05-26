package com.jsl.nlp.annotators.pos.perceptron

/**
  * Created by Saif Addin on 5/20/2017.
  */
case class Sentence(sentence: String) {
  /**
    * ToDo: Analyze whether we can re-use any tokenizer from annotators
    * @return
    */
  private val tokenRegex = "\\W+"
  def tokenize: Array[String] = {
    sentence.split(tokenRegex)
  }
}
