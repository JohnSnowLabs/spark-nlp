package com.jsl.nlp.annotators.sbd.pragmatic

import PragmaticSymbols._
import PragmaticDictionaries._

import scala.util.matching.Regex

/**
  * Created by Saif Addin on 5/6/2017.
  */
class PragmaticContentCleaner(text: String) {
  private var wip: String = text

  /**
    * Find simple lists
    * regex should match entire enumeration
    * prepend separation symbol
    * @return
    */
  def formatLists: this.type = {
    val lowerDots = "^[a-z]\\.|(?<=\\s)[a-z]\\.".r
    val lowerParens = "(\\()[a-z]+\\)|^[a-z]+\\)|\\s[a-z]+\\)".r
    val numDots = ("\\s\\d{1,2}\\.\\s|^\\d{1,2}\\.\\s|\\s\\d{1,2}\\.\\)|" +
      "^\\d{1,2}\\.\\)|\\s\\-\\d{1,2}\\.\\s|^\\-\\d{1,2}\\.\\s|" +
      "s\\-\\d{1,2}\\.\\)|^\\-\\d{1,2}(.\\))").r

    val rules = Seq(lowerDots, lowerParens, numDots)
    wip = rules.foldRight(wip)((rule, w) => rule replaceAllIn(w, _ => LIST_INDICATOR + "$0"))
    this
  }

  /**
    * Find abbreviations in non sentence breaks
    * regex should match escape character
    * replace with non separation symbol
    * @return
    */
  def formatAbbreviations: this.type = {
    // http://rubular.com/r/yqa4Rit8EY
    val possessive = "\\.(?='s\\s)|\\.(?='s$)|\\.(?='s\\z)".r
    // http://rubular.com/r/NEv265G2X2
    val kommandit = "(?<=Co)\\.(?=\\sKG)".r
    // http://rubular.com/r/e3H6kwnr6H
    val singleLetter = "(?<=^[A-Z])\\.(?=\\s)".r
    // http://rubular.com/r/gitvf0YWH4
    val singleUpper = "(?<=\\s[A-Z])\\.(?=\\s)".r
    val prepositive = PREPOSITIVE_ABBREVIATIONS.map(abbr => s"(?<=\\s$abbr)\\.(?=\\s)|(?<=^$abbr)\\.(?=\\s)".r)
    val prepositiveTagged = PREPOSITIVE_ABBREVIATIONS.map(abbr => s"(?<=\\s$abbr)\\.(?=:\\d+)|(?<=^$abbr)\\.(?=:\\d+)".r)
    val number = NUMBER_ABBREVIATIONS.map(abbr => s"(?<=\\s$abbr)\\.(?=\\s\\d)|(?<=^$abbr)\\.(?=\\s\\d)".r)
    val numberTagged = NUMBER_ABBREVIATIONS.map(abbr => s"(?<=\\s$abbr)\\.(?=\\s+\\()|(?<=^$abbr)\\.(?=\\s+\\()".r)
    val general = ABBREVIATIONS.map(abbr => (
      s"(?<=\\s$abbr)\\.(?=((\\.|\\:|-|\\?)|(\\s([a-z]|I\\s|I'm|I'll" +
      s"|\\d))))|(?<=^$abbr)\\.(?=((\\.|\\:|\\?)" +
      s"|(\\s([a-z]|I\\s|I'm|I'll|\\d))))"
      ).r)
    val generalComma = ABBREVIATIONS.map(abbr => s"(?<=\\s$abbr)\\.(?=,)|(?<=^$abbr)\\.(?=,)".r)
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

    val rules: Seq[Seq[Regex]] = Seq(
      Seq(possessive), Seq(kommandit), Seq(singleLetter), Seq(singleUpper),
      prepositive, prepositiveTagged, number, numberTagged, general, generalComma
    )

    wip = rules.flatten.foldRight(wip)((rule, w) => rule replaceAllIn(w, _ => s"$ABBREVIATOR"))

    this
  }

  /**
    * Find numbers in non sentence breaks
    * regex should match escape character
    * replace with non separation symbol
    * @return
    */
  def formatNumbers: this.type = {
    // http://rubular.com/r/oNyxBOqbyy
    val periodBefore = "\\.(?=\\d)".r
    // http://rubular.com/r/EMk5MpiUzt
    val afterPeriodBeforeLetter = "(?<=\\d)\\.(?=\\S)".r
    // http://rubular.com/r/rf4l1HjtjG
    // ToDo: To be added. Need to confirm the expected behavior
    // val newLinePeriod = "(?<=\\r\d)\.(?=(\s\S)|\))".r
    // ----
    // http://rubular.com/r/HPa4sdc6b9
    val startLinePeriod = "(?<=^\\d)\\.(?=(\\s\\S)|\\))".r
    // http://rubular.com/r/NuvWnKleFl
    val startLineTwoDigit = "(?<=^\\d\\d)\\.(?=(\\s\\S)|\\))".r

    val rules = Seq(periodBefore, afterPeriodBeforeLetter, startLinePeriod, startLineTwoDigit)

    wip = rules.foldRight(wip)((rule, w) => rule replaceAllIn(w, _ => s"$NUM_INDICATOR"))

    this
  }

  /**
    * Find sentence breaking symbols
    * regex should match entire symbol
    * append end breaking symbol
    * @return
    */
  def formatPunctuations: this.type = {
    //http://rubular.com/r/mQ8Es9bxtk
    val continuous = "(?<=\\S)(!|\\?){3,}(?=(\\s|\\z|$))".r

    val rules = Seq(continuous)

    wip = rules.foldRight(wip)((rule, w) => rule replaceAllIn(w, _ => "$0" + PUNCT_INDICATOR))

    this
  }

  /**
    * Find sentence multiple non-breaking character
    * regex should match group 2 as symbol
    * replace with non breaking symbol
    * @return
    */
  def formatMultiplePeriods: this.type = {
    // http://rubular.com/r/EUbZCNfgei
    val periods = new Regex("(\\w)(\\.)(\\w)", "pre", "per", "suf")

    val rules = Seq(periods)

    wip = rules.foldRight(wip)((rule, w) => rule replaceAllIn(w, m => s"${m group "pre"}$MULT_PERIOD${m group "suf"}"))

    this
  }

  /**
    * Find specific coordinates non-breaking characters
    * regex should match non breaking symbol
    * replace with non breaking symbol
    * @return
    */
  def formatGeoLocations: this.type = {
    // http://rubular.com/r/G2opjedIm9
    val specialPeriods = "http://rubular.com/r/G2opjedIm9".r

    val rules = Seq(specialPeriods)

    wip = rules.foldRight(wip)((rule, w) => rule replaceAllIn(w, _ => s"$PUNCT_INDICATOR"))

    this
  }

  def finish: String = wip
}
