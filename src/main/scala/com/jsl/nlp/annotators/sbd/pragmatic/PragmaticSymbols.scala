package com.jsl.nlp.annotators.sbd.pragmatic

/**
  * Created by Saif Addin on 5/6/2017.
  */
object PragmaticSymbols {

  /**
    * Basic sentence separator
    */
  val BASIC_DOT = "\\."

  /**
    * Separation symbols for list items and numbers
    * alt 197
    */
  val LIST_INDICATOR = "┼"

  /**
    * Non separation dots for abbreviations
    * alt 198
    */
  val ABBREVIATOR = "╞"

  /**
    * Non separation dots for numbers
    * alt 199
    */
  val NUM_INDICATOR = "╟"

  /**
    * Punctuation line breaker
    * alt 200
    */
  val PUNCT_INDICATOR = "╚"

  /**
    * Period container non breaker
    * alt 201
    */
  val MULT_PERIOD = "╔"

  /**
    * Special non breaking symbol
    * alt 202
    */
  val SPECIAL_PERIOD = "╩"

  /**
    * Final sentence breaking symbol
    * alt 401
    */
  val FINAL_BREAK = "æ"

  /**
    * Final sentence non-breaking symbol
    * alt 402
    */
  val FINAL_NON_BREAK = "Æ"

  def getSentenceBreakers = Seq(BASIC_DOT, LIST_INDICATOR, PUNCT_INDICATOR)

  def getSentenceNonBreakers = Seq(ABBREVIATOR, NUM_INDICATOR, MULT_PERIOD, SPECIAL_PERIOD)

}
