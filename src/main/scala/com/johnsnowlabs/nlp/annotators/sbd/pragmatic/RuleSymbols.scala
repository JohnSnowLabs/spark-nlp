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
    * 
    */
  val BREAK_INDICATOR = "\uF050"

  /**
    * looks up .
    */
  val DOT = "\uF051"

  /**
    * looks up ,
    */
  val COMMA = "\uF052"

  /**
    * looks up ;
    */
  val SEMICOLON = "\uF053"

  /**
    * looks up ?
    */
  val QUESTION = "\uF054"

  /**
    * looks up !
    */
  val EXCLAMATION = "\uF055"

  /**
    * ====================
    * PROTECTION SYMBOL
    * ====================
    */

  /**
    * Between punctuations marker
    */
  val PROTECTION_MARKER_OPEN = "\uF056"
  val PROTECTION_MARKER_CLOSE = "\uF057"

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
