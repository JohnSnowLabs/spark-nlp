package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.PragmaticDictionaries._
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.PragmaticSymbols._
import com.johnsnowlabs.nlp.util.regex.{RegexRule, RuleFactory, TransformStrategy, MatchStrategy}

/**
  * Created by Saif Addin on 5/6/2017.
  */

/**
  * rule-based formatter that adds regex rules to different marking steps
  * Symbols protect from ambiguous bounds to be considered splitters
  * @param text text to tag, which is modified in place with Symbols
  */
class PragmaticContentFormatter(text: String) {

  import TransformStrategy._
  import MatchStrategy._

  private var wip: String = text

  /**
    * Arbitrarely mark bounds with user provided characters
    * @return
    */
  def formatCustomBounds(bounds: Array[String]): this.type = {

    val factory = new RuleFactory(MATCH_ALL, REPLACE_ALL_WITH_SYMBOL)
    bounds.foreach(bound => factory.addRule(bound.r, s"split bound: $bound"))

    wip = factory.transformWithSymbol(BREAK_INDICATOR, wip)

    this
  }

  /**
    * Find simple lists
    * regex should match entire enumeration
    * prepend separation symbol
    * @return
    */
  def formatLists: this.type = {

    val factory = new RuleFactory(MATCH_ALL, PREPEND_WITH_SYMBOL)
    // http://rubular.com/r/XcpaJKH0sz
      //lower case dots
      // ToDo: This rule requires more complex logic than just itself
      //.addRule("(?<=^)[a-z]\\.|(?<=\\s)[a-z]\\.".r)
    // http://rubular.com/r/Gu5rQapywf
      //lower case parens
      .addRule(new RegexRule("(\\()[a-z]+\\)|^[a-z]+\\)", "formatLists"))
      //numeric dots
      .addRule(new RegexRule(
        "\\s\\d{1,2}\\.\\s|^\\d{1,2}\\.\\s|\\s\\d{1,2}\\.\\)|" +
        "^\\d{1,2}\\.\\)|\\s\\-\\d{1,2}\\.\\s|^\\-\\d{1,2}\\.\\s|" +
        "s\\-\\d{1,2}\\.\\)|^\\-\\d{1,2}(.\\))",
        "formatLists-numerical"
    ))

    wip = factory.transformWithSymbol(BREAK_INDICATOR, wip)

    this
  }

  /**
    * Find abbreviations in non sentence breaks
    * regex should match escape character
    * replace with non separation symbol
    * @return
    */
  def formatAbbreviations(useDictAbbreviations: Boolean): this.type = {

    val stdAbbrFactory = new RuleFactory(MATCH_ALL, REPLACE_ALL_WITH_SYMBOL)
    // http://rubular.com/r/yqa4Rit8EY
      //possessive
      .addRule(new RegexRule("\\.(?='s\\s)|\\.(?='s\\$)|\\.(?='s\\z)", "formatAbbreviations-possessive"))
    // http://rubular.com/r/NEv265G2X2
      //kommandit
      .addRule(new RegexRule("(?<=Co)\\.(?=\\sKG)", "formatAbbreviations-kommandit"))
    // http://rubular.com/r/e3H6kwnr6H
      //single letter abbreviation
      .addRule(new RegexRule("(?<=^[A-Z])\\.(?=\\s)", "formatAbbreviations-lowercaseAbb"))
    // http://rubular.com/r/gitvf0YWH4
      //single upper case letter abbreviation
      .addRule(new RegexRule("(?<=\\s[A-Z])\\.(?=\\s)", "formatAbbreviations-uppercaseAbb"))

    val specialAbbrFactory = new RuleFactory(MATCH_ALL, PROTECT_FROM_BREAK)
    // http://rubular.com/r/xDkpFZ0EgH
    // http://rubular.com/r/ezFi9y2Q1t
      //multiple period words
      .addRule(new RegexRule("\\b[a-zA-Z](?:\\.[a-zA-Z])+(?:\\.(?!\\s[A-Z]))*", "protectAbbreviations-multiplePeriod"))
    // http://rubular.com/r/Vnx3m4Spc8
      //AM PM Rules
      .addRule(new RegexRule("(?i)p\\.m\\.*", "protectAbbreviations-pm"))
      .addRule(new RegexRule("(?i)a\\.m\\.*", "protectAbbreviations-am"))

    wip = specialAbbrFactory.transformWithSymbolicRules(wip)
    wip = stdAbbrFactory.transformWithSymbol(ABBREVIATOR, wip)

    if (useDictAbbreviations) {
      val dictAbbrFactory = new RuleFactory(MATCH_ALL, REPLACE_ALL_WITH_SYMBOL)
        //prepositive
        .addRules(PREPOSITIVE_ABBREVIATIONS.map(abbr => new RegexRule(s"(?<=\\s(?i)$abbr)\\.(?=\\s)|(?<=^(?i)$abbr)\\.(?=\\s)", "formatAbbreviations-preposAbbr")))
        //tagged prepositive
        .addRules(PREPOSITIVE_ABBREVIATIONS.map(abbr => new RegexRule(s"(?<=\\s(?i)$abbr)\\.(?=:\\d+)|(?<=^(?i)$abbr)\\.(?=:\\d+)", "formatAbbreviations-preposAbbr")))
        //number abbreviation
        .addRules(NUMBER_ABBREVIATIONS.map(abbr => new RegexRule(s"(?<=\\s(?i)$abbr)\\.(?=\\s\\d)|(?<=^(?i)$abbr)\\.(?=\\s\\d)", "formatAbbreviations-numberAbbr")))
        //tagged number abbreviation
        .addRules(NUMBER_ABBREVIATIONS.map(abbr => new RegexRule(s"(?<=\\s(?i)$abbr)\\.(?=\\s+\\()|(?<=^(?i)$abbr)\\.(?=\\s+\\()", "formatAbbreviations-numberAbbr")))
        //general abbreviation
        .addRules(ABBREVIATIONS.map(abbr => new RegexRule(
        s"(?<=\\s(?i)$abbr)\\.(?=((\\.|\\:|-|\\?)|(\\s([a-z]|I\\s|I'm|I'll" +
          s"|\\d))))|(?<=^(?i)$abbr)\\.(?=((\\.|\\:|\\?)" +
          s"|(\\s([a-z]|I\\s|I'm|I'll|\\d))))"
        , "formatAbbreviations-generalAbbr")))
        //general comma abbreviation
        .addRules(ABBREVIATIONS.map(abbr => new RegexRule(s"(?<=\\s(?i)$abbr)\\.(?=,)|(?<=^(?i)$abbr)\\.(?=,)", "formatAbbreviations-otherAbbr")))
      wip = dictAbbrFactory.transformWithSymbol(ABBREVIATOR, wip)
    }

    this
  }

  /**
    * Find numbers in non sentence breaks
    * regex should match escape character
    * replace with non separation symbol
    * @return
    */
  def formatNumbers: this.type = {

    val factory = new RuleFactory(MATCH_ALL, REPLACE_ALL_WITH_SYMBOL)
    // http://rubular.com/r/oNyxBOqbyy
      //period before
      .addRule(new RegexRule("\\.(?=\\d)", "formatNumbers-periodBefore"))
    // http://rubular.com/r/EMk5MpiUzt
      //after period and before letter
      .addRule(new RegexRule("(?<=\\d)\\.(?=\\S)", "formatNumbers-periodBeforeAndBeforeLetter"))
    // http://rubular.com/r/rf4l1HjtjG
    // ToDo: To be added. Need to confirm the expected behavior
    // val newLinePeriod = "(?<=\\r\d)\.(?=(\s\S)|\))"
    // ----
    // http://rubular.com/r/HPa4sdc6b9
      //start line period
      .addRule(new RegexRule("(?<=^\\d)\\.(?=(\\s\\S)|\\))", "formatNumbers-startLinePeriod"))
    // http://rubular.com/r/NuvWnKleFl
      //start line with two digits
      .addRule(new RegexRule("(?<=^\\d\\d)\\.(?=(\\s\\S)|\\))", "formatNumbers-startLineTwoDigits"))

    wip = factory.transformWithSymbol(NUM_INDICATOR, wip)

    this
  }

  /**
    * Find sentence breaking symbols
    * regex should match entire symbol
    * append end breaking symbol
    * @return
    */
  def formatPunctuations: this.type = {

    val factory = new RuleFactory(MATCH_ALL, APPEND_WITH_SYMBOL)
    // http://rubular.com/r/iVOnFrGK6H
      //continuous punctuations
      .addRule(new RegexRule("(?<=\\S)[!\\?]+(?=\\s|\\z|\\$)", "formatPunctuations-continuous"))

    wip = factory.transformWithSymbol(BREAK_INDICATOR, wip)

    this
  }

  /**
    * Find sentence multiple non-breaking character
    * regex should match group 2 as symbol
    * replace with symbol
    * @return
    */
  def formatMultiplePeriods: this.type = {

    val factory = new RuleFactory(MATCH_ALL, REPLACE_ALL_WITH_SYMBOL)
    // http://rubular.com/r/EUbZCNfgei
      //periods
      .addRule(new RegexRule("(?<=\\w)\\.(?=\\w)", "formatMultiplePeriods"))

    wip = factory.transformWithSymbol(MULT_PERIOD, wip)

    this
  }

  /**
    * Find specific coordinates non-breaking characters
    * regex should match non breaking symbol
    * replace with non breaking symbol
    * @return
    */
  def formatGeoLocations: this.type = {
    val factory = new RuleFactory(MATCH_ALL, REPLACE_ALL_WITH_SYMBOL)
    // http://rubular.com/r/G2opjedIm9
      //special periods
      .addRule(new RegexRule("(?<=[a-zA-z]°)\\.(?=\\s*\\d+)", "formatGeo"))

    wip = factory.transformWithSymbol(MULT_PERIOD, wip)

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

    val factory = new RuleFactory(MATCH_ALL, REPLACE_WITH_SYMBOL_AND_BREAK)
    // http://rubular.com/r/i60hCK81fz
      //three consecutive
      .addRule(new RegexRule("\\.\\.\\.(?=\\s+[A-Z])", "formatEllipsis-threeConsec"))
    // http://rubular.com/r/Hdqpd90owl
      //four consecutve
      .addRule(new RegexRule("(?<=\\S)\\.{3}(?=\\.\\s[A-Z])", "formatEllipsis-fourConsec"))
    // http://rubular.com/r/2VvZ8wRbd8
    // ToDo: NOT ADDING THIS ONE FOR NOW...
    // http://rubular.com/r/2VvZ8wRbd8
      //three other rule
      //.addRule(new RegexRule("\\.\\.\\.".r, "formatEllipsis-threeOther"))

    wip = factory.transformWithSymbol(ELLIPSIS_INDICATOR, wip)

    this
  }

  /**
    * Find punctuation rules NON-BREAKING characters
    * regex should match entire wrapped sentence
    * protect entire sentence
    * @return
    */
  def formatBetweenPunctuations: this.type = {
    val factory = new RuleFactory(MATCH_ALL, PROTECT_FROM_BREAK)
    // ToDo: NOT ADDING EXCLAMATION WORDS,
    // https://github.com/diasks2/pragmatic_segmenter/blob/master/lib/pragmatic_segmenter/exclamation_words.rb

    // http://rubular.com/r/2YFrKWQUYi
      //between single quotes
      .addRule(new RegexRule("'[\\w\\s?!\\.,']+'", "betweenPunctuations-singleQuot"))
    // http://rubular.com/r/3Pw1QlXOjd
      //between double quotes
      .addRule(new RegexRule("\"[\\w\\s?!\\.,]+\"", "betweenPunctuations-doubleQuot"))
    // http://rubular.com/r/WX4AvnZvlX
      //between square brackets
      .addRule(new RegexRule("\\[[\\w\\s?!\\.,]+\\]", "betweenPunctuations-squareBrack"))
    // http://rubular.com/r/6tTityPflI
      //between parens
      .addRule(new RegexRule("\\([\\w\\s?!\\.,]+\\)", "betweenPunctuations-parens"))
    wip = factory.transformWithSymbolicRules(wip)

    this
  }

  /**
    * Specific case for question mark in quotes
    * regex should match question mark
    * replace with symbol
    * @return
    */
  def formatQuotationMarkInQuotation: this.type = {

    val factory = new RuleFactory(MATCH_ALL, REPLACE_ALL_WITH_SYMBOL)
    //http://rubular.com/r/aXPUGm6fQh
      //question mark in quotes
      .addRule(new RegexRule("\\?(?=(\\'|\\\"))", "quotationMarkInQuot"))

    wip = factory.transformWithSymbol(QUESTION_IN_QUOTE, wip)

    this
  }

  /**
    * Specific cases for exclamation marks
    * regex should match exclamation mark
    * replace with symbol
    * @return
    */
  def formatExclamationPoint: this.type = {

    val factory = new RuleFactory(MATCH_ALL, REPLACE_ALL_WITH_SYMBOL)
    // http://rubular.com/r/XS1XXFRfM2
      //in quote
      .addRule(new RegexRule("\\!(?=(\\'|\\\"))", "exclamationPoint-inQuot"))
    // http://rubular.com/r/sl57YI8LkA
      //before comma
      .addRule(new RegexRule("\\!(?=\\,\\s[a-z])", "exclamationPoint-beforeComma"))
    // http://rubular.com/r/f9zTjmkIPb
      //mid sentence
      .addRule(new RegexRule("\\!(?=\\s[a-z])", "exclamationPoint-midSentence"))

    wip = factory.transformWithSymbol(EXCLAMATION_INDICATOR, wip)

    this
  }

  def formatBasicBreakers: this.type = {
    val factory = new RuleFactory(MATCH_ALL, REPLACE_EACH_WITH_SYMBOL_AND_BREAK)
      .addSymbolicRule(DOT, new RegexRule("\\.", "basicBreakers-dot"))
      .addSymbolicRule(SEMICOLON, new RegexRule(";", "basicBreakers-semicolon"))

    wip = factory.transformWithSymbolicRules(wip)

    this
  }

  /**
    * ToDo: NOT DOING replace_parens IN LISTS
    * @return
    */

  def finish: String = wip
}
