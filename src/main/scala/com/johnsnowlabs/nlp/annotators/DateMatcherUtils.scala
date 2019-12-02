package com.johnsnowlabs.nlp.annotators

import java.util.Calendar

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RuleFactory}
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, Params}

import scala.util.matching.Regex

trait DateMatcherUtils extends Params {

  /**
    * Container of a parsed date with identified bounds
    * @param calendar [[Calendar]] holding parsed date
    * @param start start bound of detected match
    * @param end end bound of detected match
    */
  private[annotators] case class MatchedDateTime(calendar: Calendar, start: Int, end: Int)

  /** Standard formal dates, e.g. 05/17/2014 or 17/05/2014 or 2014/05/17 */
  private val formalDate = new Regex("\\b(0?[1-9]|1[012])[-/]([0-2]?[1-9]|[1-3][0-1])[-/](\\d{2,4})\\b", "month", "day", "year")
  private val formalDateAlt = new Regex("\\b([0-2]?[1-9]|[1-3][0-1])[-/](0?[1-9]|1[012])[-/](\\d{2,4})\\b", "day", "month", "year")
  private val formalDateAlt2 = new Regex("\\b(\\d{2,4})[-/](0?[1-9]|1[012])[-/]([0-2]?[1-9]|[1-3][0-1])\\b", "year", "month", "day")
  private val formalDateShort = new Regex("\\b(0?[1-9]|1[012])[-/](\\d{2,4})\\b", "month", "year")

  private val months = Seq("january","february","march","april","may","june","july","august","september","october","november","december")
  protected val shortMonths = Seq("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec")

  /** Relaxed dates, e.g. March 2nd */
  private val relaxedDayNumbered = "\\b([0-2]?[1-9]|[1-3][0-1])(?:st|rd|nd|th)*\\b".r
  private val relaxedMonths = "(?i)" + months.zip(shortMonths).map(m => m._1 + "|" + m._2).mkString("|")
  private val relaxedYear = "\\d{4}\\b|\\B'\\d{2}\\b".r

  /** Relative dates, e.g. tomorrow */
  private val relativeDate = "(?i)(next|last)\\s(week|month|year)".r
  private val relativeDay = "(?i)(today|tomorrow|yesterday|past tomorrow|day before|day after|day before yesterday|day after tomorrow)".r
  private val relativeExactDay = "(?i)(next|last|past)\\s(mon|tue|wed|thu|fri)".r

  /** standard time representations e.g. 05:42:16 or 5am*/
  private val clockTime = new Regex("(?i)([0-2][0-9]):([0-5][0-9])(?::([0-5][0-9]))?", "hour", "minutes", "seconds")
  private val altTime = new Regex("([0-2]?[0-9])\\.([0-5][0-9])\\.?([0-5][0-9])?", "hour", "minutes", "seconds")
  private val coordTIme = new Regex("([0-2]?[0-9])([0-5][0-9])?\\.?([0-5][0-9])?\\s*(?:h|a\\.?m|p\\.?m)", "hour", "minutes", "seconds")
  private val refTime = new Regex("at\\s+([0-9])\\s*([0-5][0-9])*\\s*([0-5][0-9])*")
  protected val amDefinition = "(?i)(a\\.?m)".r

  /** Annotator param containing expected output format of parsed date*/
  val dateFormat: Param[String] = new Param(this, "dateFormat", "SimpleDateFormat standard criteria")

  def getFormat: String = $(dateFormat)

  def setFormat(value: String): this.type = set(dateFormat, value)

  val readMonthFirst: BooleanParam = new BooleanParam(this, "readMonthFirst", "Whether to parse july 07/05/2015 or as 05/07/2015")

  def setReadMonthFirst(value: Boolean): this.type = set(readMonthFirst, value)

  def getReadMonthFirst: Boolean = $(readMonthFirst)

  val defaultDayWhenMissing: IntParam = new IntParam(this, "defaultDayWhenMissing", "which day to set when it is missing from parsed input")

  def setDefaultDayWhenMissing(value: Int): this.type = set(defaultDayWhenMissing, value)

  def getDefaultDayWhenMissing: Int = $(defaultDayWhenMissing)

  setDefault(
    dateFormat -> "yyyy/MM/dd",
    readMonthFirst -> true,
    defaultDayWhenMissing -> 1
  )

  /**
    * Searches formal date by ordered rules
    * Matching strategy is to find first match only, ignore additional matches from then
    * Any 4 digit year will be assumed a year, any 2 digit year will be as part of XX Century e.g. 1954
    */
  protected val formalFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)

  if ($(readMonthFirst))
    formalFactory
      .addRule(formalDate, "formal date with month at first")
      .addRule(formalDateAlt, "formal date with day at first")
      .addRule(formalDateAlt2, "formal date with year at beginning")
      .addRule(formalDateShort, "formal date short version")
  else
    formalFactory
      .addRule(formalDateAlt, "formal date with day at first")
      .addRule(formalDate, "formal date with month at first")
      .addRule(formalDateAlt2, "formal date with year at beginning")
      .addRule(formalDateShort, "formal date short version")

  /**
    * Searches relaxed dates by ordered rules by more exhaustive to less
    * Strategy used is to match first only. any other matches discarded
    * Auto completes short versions of months. Any two digit year is considered to be XX century
    */
  protected val relaxedFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    .addRule(relaxedDayNumbered, "relaxed days")
    .addRule(relaxedMonths.r, "relaxed months exclusive")
    .addRule(relaxedYear, "relaxed year")

  /**
    * extracts relative dates. Strategy is to get only first match.
    * Will always assume relative day from current time at processing
    * ToDo: Support relative dates from input date
    */
  protected val relativeFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    .addRule(relativeDate, "relative dates")

  /** Searches for relative informal dates such as today or the day after tomorrow */
  protected val tyFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    .addRule(relativeDay, "relative days")

  /** Searches for exactly provided days of the week. Always relative from current time at processing */
  protected val relativeExactFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    .addRule(relativeExactDay, "relative precise dates")

  /**
    * Searches for times of the day
    * dateTime If any dates found previously, keep it as part of the final result
    * text target document
    * @return a final possible date if any found
    */
  protected val timeFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    .addRule(clockTime, "standard time extraction")
    .addRule(altTime, "alternative time format")
    .addRule(coordTIme, "coordinate like time")
    .addRule(refTime, "referred time")

  protected def formalDateContentParse(date: RuleFactory.RuleMatch): MatchedDateTime = {
    val formalDate = date.content
    val calendar = new Calendar.Builder()
    MatchedDateTime(
      calendar.setDate(
        if (formalDate.group("year").toInt > 999)
          formalDate.group("year").toInt
        /** If year found is greater than <10> years from now, assume text is talking about 20th century */
        else if (formalDate.group("year").toInt > Calendar.getInstance.get(Calendar.YEAR).toString.takeRight(2).toInt + 10)
          formalDate.group("year").toInt + 1900
        else
          formalDate.group("year").toInt + 2000,
        formalDate.group("month").toInt - 1,
        if (formalDate.groupCount == 3) formalDate.group("day").toInt else $(defaultDayWhenMissing)
      ).build(),
      formalDate.start,
      formalDate.end
    )
  }

  protected def relativeDateContentParse(date: RuleFactory.RuleMatch): MatchedDateTime = {
    val relativeDate = date.content
    val calendar = Calendar.getInstance()
    val amount = if (relativeDate.group(1) == "next") 1 else -1
    relativeDate.group(2) match {
      case "week" => calendar.add(Calendar.WEEK_OF_MONTH, amount)
      case "month" => calendar.add(Calendar.MONTH, amount)
      case "year" => calendar.add(Calendar.YEAR, amount)
    }
    MatchedDateTime(calendar, relativeDate.start, relativeDate.end)
  }

  def tomorrowYesterdayContentParse(date: RuleFactory.RuleMatch): MatchedDateTime = {
    val tyDate = date.content
    tyDate.matched.toLowerCase match {
      case "today" =>
        val calendar = Calendar.getInstance()
        MatchedDateTime(calendar, tyDate.start, tyDate.end)
      case "tomorrow" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, 1)
        MatchedDateTime(calendar, tyDate.start, tyDate.end)
      case "past tomorrow" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, 2)
        MatchedDateTime(calendar, tyDate.start, tyDate.end)
      case "yesterday" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, -1)
        MatchedDateTime(calendar, tyDate.start, tyDate.end)
      case "day after" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, 1)
        MatchedDateTime(calendar, tyDate.start, tyDate.end)
      case "day before" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, -1)
        MatchedDateTime(calendar, tyDate.start, tyDate.end)
      case "day after tomorrow" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, 2)
        MatchedDateTime(calendar, tyDate.start, tyDate.end)
      case "day before yesterday" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, -2)
        MatchedDateTime(calendar, tyDate.start, tyDate.end)
    }
  }

  def relativeExactContentParse(possibleDate: RuleFactory.RuleMatch): MatchedDateTime = {
    val relativeDate = possibleDate.content
    val calendar = Calendar.getInstance()
    val amount = if (relativeDate.group(1) == "next") 1 else -1
    calendar.add(Calendar.DAY_OF_MONTH, amount)
    relativeDate.group(2) match {
      case "mon" =>
        while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.MONDAY) {
          calendar.add(Calendar.DAY_OF_MONTH, amount)
        }
      case "tue" =>
        while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.TUESDAY) {
          calendar.add(Calendar.DAY_OF_MONTH, amount)
        }
      case "wed" =>
        while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.WEDNESDAY) {
          calendar.add(Calendar.DAY_OF_MONTH, amount)
        }
      case "thu" =>
        while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.THURSDAY) {
          calendar.add(Calendar.DAY_OF_MONTH, amount)
        }
      case "fri" =>
        while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.FRIDAY) {
          calendar.add(Calendar.DAY_OF_MONTH, amount)
        }
    }
    MatchedDateTime(calendar, relativeDate.start, relativeDate.end)
  }

}
