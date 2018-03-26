package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

/**
  * Created by Saif Addin on 5/6/2017.
  */

/**
  * Base Symbols that may be extended later on. For now kept in the pragmatic scope.
  */
trait RuleSymbols {

  /**
    * Separation symbols for list items and numbers
    * alt 197
    */
  val BREAK_INDICATOR = "┼"

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
    // http://rubular.com/r/Tq7lWxGkQl
  val UNPROTECTED_BREAK_INDICATOR = s"$BREAK_INDICATOR(?![^$PROTECTION_MARKER_OPEN]*$PROTECTION_MARKER_CLOSE)"

  def symbolRecovery: Map[String, String] = Map(
    DOT -> ".",
    SEMICOLON -> ";",
    QUESTION -> "?",
    EXCLAMATION -> "!",
    PROTECTION_MARKER_OPEN -> "",
    PROTECTION_MARKER_CLOSE -> ""
  )

}
