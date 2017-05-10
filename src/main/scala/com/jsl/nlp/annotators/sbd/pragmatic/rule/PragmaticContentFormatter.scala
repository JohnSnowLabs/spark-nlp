package com.jsl.nlp.annotators.sbd.pragmatic.rule

import com.jsl.nlp.annotators.sbd.pragmatic.PragmaticDictionaries.{ABBREVIATIONS, NUMBER_ABBREVIATIONS, PREPOSITIVE_ABBREVIATIONS}
import com.jsl.nlp.annotators.sbd.pragmatic.PragmaticSymbols._

/**
  * Created by Saif Addin on 5/6/2017.
  */
class PragmaticContentFormatter(text: String) {
  import RuleStrategy._

  private var wip: String = text

  /**
    * Find simple lists
    * regex should match entire enumeration
    * prepend separation symbol
    * @return
    */
  def formatLists: this.type = {

    val factory = new RuleFactory(PREPEND_WITH_SYMBOL)
    // http://rubular.com/r/XcpaJKH0sz
      //lower case dots
      .addRule("(?<=^)[a-z]\\.|(?<=\\s)[a-z]\\.".r)
    // http://rubular.com/r/Gu5rQapywf
      //lower case parens
      .addRule("(\\()[a-z]+\\)|^[a-z]+\\)|\\s[a-z]+\\)".r)
      //numeric dots
      .addRule(
        ("\\s\\d{1,2}\\.\\s|^\\d{1,2}\\.\\s|\\s\\d{1,2}\\.\\)|" +
        "^\\d{1,2}\\.\\)|\\s\\-\\d{1,2}\\.\\s|^\\-\\d{1,2}\\.\\s|" +
        "s\\-\\d{1,2}\\.\\)|^\\-\\d{1,2}(.\\))").r)

    wip = factory.applyWith(BREAK_INDICATOR, wip)

    this
  }

  /**
    * Find abbreviations in non sentence breaks
    * regex should match escape character
    * replace with non separation symbol
    * @return
    */
  def formatAbbreviations: this.type = {

    val factory = new RuleFactory(REPLACE_ALL_WITH_SYMBOL)
    // http://rubular.com/r/yqa4Rit8EY
      //possessive
      .addRule("\\.(?='s\\s)|\\.(?='s$)|\\.(?='s\\z)".r)
    // http://rubular.com/r/NEv265G2X2
      //kommandit
      .addRule("(?<=Co)\\.(?=\\sKG)".r)
    // http://rubular.com/r/e3H6kwnr6H
      //single letter abbreviation
      .addRule("(?<=^[A-Z])\\.(?=\\s)".r)
    // http://rubular.com/r/gitvf0YWH4
      //single upper case letter abbreviation
      .addRule("(?<=\\s[A-Z])\\.(?=\\s)".r)
      //prepositive
      .addRules(PREPOSITIVE_ABBREVIATIONS.map(abbr => s"(?<=\\s$abbr)\\.(?=\\s)|(?<=^$abbr)\\.(?=\\s)".r))
      //tagged prepositive
      .addRules(PREPOSITIVE_ABBREVIATIONS.map(abbr => s"(?<=\\s$abbr)\\.(?=:\\d+)|(?<=^$abbr)\\.(?=:\\d+)".r))
      //number abbreviation
      .addRules(NUMBER_ABBREVIATIONS.map(abbr => s"(?<=\\s$abbr)\\.(?=\\s\\d)|(?<=^$abbr)\\.(?=\\s\\d)".r))
      //tagged number abbreviation
      .addRules(NUMBER_ABBREVIATIONS.map(abbr => s"(?<=\\s$abbr)\\.(?=\\s+\\()|(?<=^$abbr)\\.(?=\\s+\\()".r))
      //general abbreviation
      .addRules(ABBREVIATIONS.map(abbr => (
      s"(?<=\\s$abbr)\\.(?=((\\.|\\:|-|\\?)|(\\s([a-z]|I\\s|I'm|I'll" +
      s"|\\d))))|(?<=^$abbr)\\.(?=((\\.|\\:|\\?)" +
      s"|(\\s([a-z]|I\\s|I'm|I'll|\\d))))"
      ).r))
      //general comma abbreviation
      .addRules(ABBREVIATIONS.map(abbr => s"(?<=\\s$abbr)\\.(?=,)|(?<=^$abbr)\\.(?=,)".r))
    /*
    ** ToDo: requires a special treatment to protect a word such as U.S.A. and P.M.
    // http://rubular.com/r/xDkpFZ0EgH
    val multiPeriod = "\\b[a-z](?:\\.[a-z])+[.]".r
    http://rubular.com/r/Vnx3m4Spc8
    val upperAm = "(?<=P∯M)∯(?=\\s[A-Z])".r
    val upperPm = "(?<=A∯M)∯(?=\s[A-Z])".r
    val lowerAm = "(?<=p∯m)∯(?=\s[A-Z])".r
    val lowerPm = "(?<=a∯m)∯(?=\s[A-Z])".r
    */

    wip = factory.applyWith(ABBREVIATOR, wip)

    this
  }

  /**
    * Find numbers in non sentence breaks
    * regex should match escape character
    * replace with non separation symbol
    * @return
    */
  def formatNumbers: this.type = {

    val factory = new RuleFactory(REPLACE_ALL_WITH_SYMBOL)
    // http://rubular.com/r/oNyxBOqbyy
      //period before
      .addRule("\\.(?=\\d)".r)
    // http://rubular.com/r/EMk5MpiUzt
      //after period and before letter
      .addRule("(?<=\\d)\\.(?=\\S)".r)
    // http://rubular.com/r/rf4l1HjtjG
    // ToDo: To be added. Need to confirm the expected behavior
    // val newLinePeriod = "(?<=\\r\d)\.(?=(\s\S)|\))".r
    // ----
    // http://rubular.com/r/HPa4sdc6b9
      //start line period
      .addRule("(?<=^\\d)\\.(?=(\\s\\S)|\\))".r)
    // http://rubular.com/r/NuvWnKleFl
      //start line with two digits
      .addRule("(?<=^\\d\\d)\\.(?=(\\s\\S)|\\))".r)

    wip = factory.applyWith(NUM_INDICATOR, wip)

    this
  }

  /**
    * Find sentence breaking symbols
    * regex should match entire symbol
    * append end breaking symbol
    * @return
    */
  def formatPunctuations: this.type = {

    val factory = new RuleFactory(PROTECT_WITH_SYMBOL)
    // http://rubular.com/r/mQ8Es9bxtk
      //continuous punctuations
      .addRule("(?<=\\S)(!|\\?){3,}(?=(\\s|\\z|$))".r)

    wip = factory.applyWith(PROTECTION_MARKER, wip)

    this
  }

  /**
    * Find sentence multiple non-breaking character
    * regex should match group 2 as symbol
    * replace with symbol
    * @return
    */
  def formatMultiplePeriods: this.type = {

    val factory = new RuleFactory(REPLACE_ALL_WITH_SYMBOL)
    // http://rubular.com/r/EUbZCNfgei
      //periods
      .addRule("(?<=\\w)(\\.)(?=\\w)".r)

    wip = factory.applyWith(MULT_PERIOD, wip)

    this
  }

  /**
    * Find specific coordinates non-breaking characters
    * regex should match non breaking symbol
    * replace with non breaking symbol
    * @return
    */
  def formatGeoLocations: this.type = {
    val factory = new RuleFactory(REPLACE_ALL_WITH_SYMBOL)
    // http://rubular.com/r/G2opjedIm9
      //special periods
      .addRule("http://rubular.com/r/G2opjedIm9".r)

    wip = factory.applyWith(MULT_PERIOD, wip)

    this
  }

  /**
    * WHY DOES HE DO THIS CHECK? Look for: PARENS_BETWEEN_DOUBLE_QUOTES_REGEX
  def formatParensBetweenQuotes: this.type = {
    // http://rubular.com/r/6flGnUMEVl
    val parensBQuotes = "[\"”]\\s\\(.*\\)\\s[\"“]".r
    ...
  }(
    */

  /**
    * Find ellipsis BREAKING characters WITH REPLACEMENT
    * regex should match the ellipsis
    * replace with non breaking symbol
    * @return
    */
  def formatEllipsisRules: this.type = {

    val factory = new RuleFactory(REPLACE_WITH_SYMBOL_AND_BREAK)
    // http://rubular.com/r/i60hCK81fz
      //three consecutive
      .addRule("\\.\\.\\.(?=\\s+[A-Z])".r)
    // http://rubular.com/r/Hdqpd90owl
      //four consecutve
      .addRule("(?<=\\S)\\.{3}(?=\\.\\s[A-Z])".r)
    // http://rubular.com/r/2VvZ8wRbd8
    // ToDo: NOT ADDING THIS ONE FOR NOW...
    // http://rubular.com/r/2VvZ8wRbd8
      //three other rule
      .addRule("\\.\\.\\.".r)

    wip = factory.applyWith(ELLIPSIS_INDICATOR, wip)

    this
  }

  /**
    * Find punctuation rules NON-BREAKING characters
    * regex should match entire wrapped sentence
    * protect entire sentence
    * @return
    */
  def formatBetweenPunctuations: this.type = {

    val factory = new RuleFactory(PROTECT_WITH_SYMBOL)
    // ToDo: NOT ADDING EXCLAMATION WORDS,
    // https://github.com/diasks2/pragmatic_segmenter/blob/master/lib/pragmatic_segmenter/exclamation_words.rb

    // http://rubular.com/r/2YFrKWQUYi
      //between single quotes
      .addRule("(?<=\\s)'(?:[^']|'[a-zA-Z])*'".r)
    // http://rubular.com/r/3Pw1QlXOjd
      //between double quotes
      .addRule("\"(?>[^\"\\\\]+|\\\\{2}|\\\\.)*\"".r)
    // http://rubular.com/r/x6s4PZK8jc
      //between arrow quotes
      .addRule("«(?>[^»\\\\]+|\\\\{2}|\\\\.)*»".r)
    // http://rubular.com/r/JbAIpKdlSq
      //between slant quotes
      .addRule("“(?>[^”\\\\]+|\\\\{2}|\\\\.)*”".r)
    // http://rubular.com/r/WX4AvnZvlX
      //between square brackets
      .addRule("\\[(?>[^\\]\\\\]+|\\\\{2}|\\\\.)*\\]".r)
    // http://rubular.com/r/6tTityPflI
      //between parens
      .addRule("\\((?>[^\\(\\)\\\\]+|\\\\{2}|\\\\.)*\\)".r)
    // http://rubular.com/r/mXf8cW025o
      //between leading apostrophes
      .addRule("(?<=\\s)'(?:[^']|'[a-zA-Z])*'\\S".r)

    wip = factory.applyWith(PROTECTION_MARKER, wip)

    this
  }

  /**
    * Find double punctuation BREAKING characters WITH REPLACEMENT
    * regex should match punctuations
    * replace with symbol
    * @return
    */
  def formatDoublePunctuations: this.type = {

    val factory = new RuleFactory(REPLACE_EACH_WITH_SYMBOL_AND_BREAK)
      .addSymbolicRule(DP_FIRST,"\\?!".r)
      .addSymbolicRule(DP_SECOND,"!\\?".r)
      .addSymbolicRule(DP_THIRD,"\\?\\?".r)
      .addSymbolicRule(DP_FOURTH,"!!".r)

    wip = factory.applySymbolicRules(wip)

    this
  }

  /**
    * Specific case for question mark in quotes
    * regex should match question mark
    * replace with symbol
    * @return
    */
  def formatQuotationMarkInQuotation: this.type = {

    val factory = new RuleFactory(REPLACE_ALL_WITH_SYMBOL)
    //http://rubular.com/r/aXPUGm6fQh
      //question mark in quotes
      .addRule("\\?(?=(\\'|\\\"))".r)

    wip = factory.applyWith(QUESTION_IN_QUOTE, wip)

    this
  }

  /**
    * Specific cases for exclamation marks
    * regex should match exclamation mark
    * replace with symbol
    * @return
    */
  def formatExclamationPoint: this.type = {

    val factory = new RuleFactory(REPLACE_ALL_WITH_SYMBOL)
    // http://rubular.com/r/XS1XXFRfM2
      //in quote
      .addRule("\\!(?=(\\'|\\\"))".r)
    // http://rubular.com/r/sl57YI8LkA
      //before comma
      .addRule("\\!(?=\\,\\s[a-z])".r)
    // http://rubular.com/r/f9zTjmkIPb
      //mid sentence
      .addRule("\\!(?=\\s[a-z])".r)

    wip = factory.applyWith(EXCLAMATION_INDICATOR, wip)

    this
  }

  def formatBasicBreakers: this.type = {
    val factory = new RuleFactory(REPLACE_EACH_WITH_SYMBOL_AND_BREAK)
      .addSymbolicRule(DOT, "\\.".r)
      .addSymbolicRule(COMMA, ",".r)
      .addSymbolicRule(SEMICOLON, ";".r)
      .addSymbolicRule(QUESTION, "\\?".r)
      .addSymbolicRule(EXCLAMATION, "!".r)

    wip = factory.applySymbolicRules(wip)

    this
  }

  /**
    * ToDo: NOT DOING replace_parens IN LISTS
    * @return
    */

  def finish: String = wip
}
