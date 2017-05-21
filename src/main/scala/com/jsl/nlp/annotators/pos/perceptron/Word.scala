package com.jsl.nlp.annotators.pos.perceptron

/**
  * Created by Saif Addin on 5/20/2017.
  */
case class Word(word: String) {
  def normalized: String = {
    if (word.contains("-") && word.head != '-') {
      "!HYPEN"
    } else if (word.forall(_.isDigit) && word.length == 4) {
      "!YEAR"
    } else if (word.head.isDigit) {
      "!DIGITS"
    } else {
      word.toLowerCase
    }
  }
}
