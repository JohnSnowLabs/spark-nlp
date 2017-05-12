package com.jsl.nlp.annotators.sbd.pragmatic

/**
  * Created by Saif Addin on 5/6/2017.
  */
object PragmaticSymbols {

  /**
    * ====================
    * BREAKING SYMBOLS
    * ====================
    */

  /**
    * looks up .
    * alt 401
    */
  val DOT = "æ"

  /**
    * looks up ,
    * alt 402
    */
  val COMMA = "Æ"

  /**
    * looks up ;
    * alt 403
    */
  val SEMICOLON = "ô"

  /**
    * looks up ?
    * alt 404
    */
  val QUESTION = "ö"

  /**
    * looks up !
    * alt 405
    */
  val EXCLAMATION = "ò"

  /**
    * Separation symbols for list items and numbers
    * alt 197
    */
  val BREAK_INDICATOR = "┼"

  /**
    * Punctuation line breaker
    * alt 200
    */
  val PUNCT_INDICATOR = "╚"

  /**
    * Ellipsis breaker
    * looks up -> "..."
    * alt 203
    */
  val ELLIPSIS_INDICATOR = "╦"

  /**
    * Double punctuations marker
    * alt 205-208
    */
  val DP_FIRST = "═" // ?!
  val DP_SECOND = "╬" // !?
  val DP_THIRD = "╧" // ??
  val DP_FOURTH = "╨" // !!

  /**
    * =====================
    * NON BREAKING SYMBOLS
    * =====================
    */

  /**
    * Non separation dots for abbreviations
    * looks up -> .
    * alt 198
    */
  val ABBREVIATOR = "╞"

  /**
    * Non separation dots for numbers
    * looks up -> .
    * alt 199
    */
  val NUM_INDICATOR = "╟"

  /**
    * Period container non breaker
    * looks up -> .
    * alt 201
    */
  val MULT_PERIOD = "╔"

  /**
    * Special non breaking symbol
    * alt 202
    */
  val SPECIAL_PERIOD = "╩"

  /**
    * Question in quotes
    * alt 209
    */
  val QUESTION_IN_QUOTE = "╤" // ?

  /**
    * Exclamation mark in rules
    * alt 210
    */
  val EXCLAMATION_INDICATOR = "╥" // !

  /**
    * ====================
    * PROTECTION SYMBOL
    * ====================
    */

  /**
    * Between punctuations marker
    * alt 505
    * alt 506
    */
  val PROTECTION_MARKER_OPEN = "∙"
  val PROTECTION_MARKER_CLOSE = "·"

  /**
    * Magic regex ensures no breaking within protection
    */
  val PROTECTED_BREAKER = s"$BREAK_INDICATOR+(?![^$PROTECTION_MARKER_OPEN]+$PROTECTION_MARKER_CLOSE)"

  val sentenceRecovery = Map(
    DOT -> ".",
    //COMMA -> ",",
    SEMICOLON -> ";",
    QUESTION -> "?",
    EXCLAMATION -> "!",
    ABBREVIATOR -> ".",
    NUM_INDICATOR -> ".",
    MULT_PERIOD -> ".",
    QUESTION_IN_QUOTE -> "?",
    EXCLAMATION_INDICATOR -> "!",
    ELLIPSIS_INDICATOR -> "...",
    DP_FIRST -> "?!",
    DP_SECOND-> "!?",
    DP_THIRD -> "??",
    DP_FOURTH -> "!!",
    PROTECTION_MARKER_OPEN -> "",
    PROTECTION_MARKER_CLOSE -> ""
  )

}
