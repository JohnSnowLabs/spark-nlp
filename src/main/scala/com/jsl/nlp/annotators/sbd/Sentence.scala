package com.jsl.nlp.annotators.sbd

/**
  * Created by Saif Addin on 5/5/2017.
  */

/**
  * structure representing a sentence and its boundaries
  */
case class Sentence(content: String, begin: Int, end: Int)
