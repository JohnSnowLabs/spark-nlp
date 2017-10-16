package com.johnsnowlabs.nlp.annotators

import java.text.SimpleDateFormat

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}

import scala.util.matching.Regex
import java.util.Calendar

import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RuleFactory}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Matches standard date formats into a provided format
  * @param uid internal uid required to generate writable annotators
  * @@ dateFormat: allows to define expected output format. Follows SimpleDateFormat standard.
  */
class DateMatcher(override val uid: String) extends AnnotatorModel[DateMatcher] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /**
    * Container of a parsed date with identified bounds
    * @param calendar [[Calendar]] holding parsed date
    * @param start start bound of detected match
    * @param end end bound of detected match
    */
  private[annotators] case class MatchedDateTime(calendar: Calendar, start: Int, end: Int)

  /** Standard formal dates, e.g. 2014/05/17 */
  private val formalDate = new Regex("(\\b\\d{2,4})[-/](\\d{1,2})[-/](\\d{1,2}\\b)", "year", "month", "day")
  private val formalDateAlt = new Regex("(\\b\\d{1,2})[-/](\\d{1,2})[-/](\\d{2,4}\\b)", "month", "day", "year")

  private val months = Seq("january","february","march","april","may","june","july","august","september","october","november","december")
  private val shortMonths = Seq("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec")

  /** Relaxed dates, e.g. March 2nd */
  private val relaxedDayNumbered = "\\b(\\d{1,2})(?:st|rd|nd|th)*\\b".r
  private val relaxedMonths = "(?i)" + months.mkString("|")
  private val relaxedShortMonths = "(?i)" + shortMonths.mkString("|")
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
  private val amDefinition = "(?i)(a\\.?m)".r

  /** Annotator param containing expected output format of parsed date*/
  val dateFormat: Param[String] = new Param(this, "dateFormat", "SimpleDateFormat standard criteria")

  override val annotatorType: AnnotatorType = DATE

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  setDefault(inputCols, Array(DOCUMENT))

  /** Internal constructor to submit a random UID */
  def this() = this(Identifiable.randomUID("DATE"))

  def getFormat: String = get(dateFormat).getOrElse("yyyy/MM/dd")

  def setFormat(value: String): this.type = set(dateFormat, value)

  /**
    * Finds dates in a specific order, from formal to more relaxed. Add time of any, or stand-alone time
    * @param text input text coming from target document
    * @return a possible date-time match
    */
  private[annotators] def extractDate(text: String): Option[MatchedDateTime] = {
    val possibleDate = extractFormalDate(text)
      .orElse(extractRelaxedDate(text))
      .orElse(extractRelativeDate(text))
      .orElse(extractTomorrowYesterday(text))
      .orElse(extractRelativeExactDay(text))

    possibleDate.orElse(setTimeIfAny(possibleDate, text))
  }

  /**
    * Searches formal date by ordered rules
    * Matching strategy is to find first match only, ignore additional matches from then
    * Any 4 digit year will be assumed a year, any 2 digit year will be as part of XX Century e.g. 1954
    */
  private def extractFormalDate(text: String): Option[MatchedDateTime] = {
    val formalFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    formalFactory.addRule(formalDate, "formal date matcher with year at first")
    formalFactory.addRule(formalDateAlt, "formal date with year at end")
    formalFactory.findMatchFirstOnly(text).map{ possibleDate =>
      val formalDate = possibleDate.content
      val calendar = new Calendar.Builder()
      MatchedDateTime(
        calendar.setDate(
          if (formalDate.group("year").toInt > 999)
            formalDate.group("year").toInt else
            formalDate.group("year").toInt + 1900,
          formalDate.group("month").toInt - 1,
          formalDate.group("day").toInt
        ).build(),
        formalDate.start,
        formalDate.end
      )
    }
  }

  /**
    * Searches relaxed dates by ordered rules by more exhaustive to less
    * Strategy used is to match first only. any other matches discarded
    * Auto completes short versions of months. Any two digit year is considered to be XX century
    */
  private def extractRelaxedDate(text: String): Option[MatchedDateTime] = {
    val relaxedFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    relaxedFactory.addRule(relaxedDayNumbered, "relaxed days")
    relaxedFactory.addRule(relaxedMonths.r, "relaxed months exclusive")
    relaxedFactory.addRule(relaxedShortMonths.r, "relaxed months abbreviated exclusive")
    relaxedFactory.addRule(relaxedYear, "relaxed year")
    val possibleDates = relaxedFactory.findMatch(text)
    if (possibleDates.length > 1) {
      val dayMatch = possibleDates.head.content
      val day = dayMatch.matched.filter(_.isDigit).toInt
      val monthMatch = possibleDates(1).content
      val month = shortMonths.indexOf(monthMatch.matched.toLowerCase.take(3))
      val yearMatch = possibleDates.last.content
      val year = {
        if (possibleDates.length > 2) {
          val number = yearMatch.matched.filter(_.isDigit).toInt
          if (number > 999) number else number + 1900
        } else {
          Calendar.getInstance.get(Calendar.YEAR)
        }
      }
      val calendar = new Calendar.Builder()
      calendar.setDate(year, month, day)
      Some(MatchedDateTime(
        calendar.build(),
        Seq(yearMatch, monthMatch, dayMatch).map(_.start).min,
        Seq(yearMatch, monthMatch, dayMatch).map(_.end).max
      ))
    } else None
  }

  /**
    * extracts relative dates. Strategy is to get only first match.
    * Will always assume relative day from current time at processing
    * ToDo: Support relative dates from input date
    */
  private def extractRelativeDate(text: String): Option[MatchedDateTime] = {
    val relativeFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    relativeFactory.addRule(relativeDate, "relative dates")
    relativeFactory.findMatchFirstOnly(text).map(possibleDate => {
      val relativeDate = possibleDate.content
      val calendar = Calendar.getInstance()
      val amount = if (relativeDate.group(1) == "next") 1 else -1
      relativeDate.group(2) match {
        case "week" => calendar.add(Calendar.WEEK_OF_MONTH, amount)
        case "month" => calendar.add(Calendar.MONTH, amount)
        case "year" => calendar.add(Calendar.YEAR, amount)
      }
      MatchedDateTime(calendar, relativeDate.start, relativeDate.end)
    })
  }

  /** Searches for relative informal dates such as today or the day after tomorrow */
  private def extractTomorrowYesterday(text: String): Option[MatchedDateTime] = {
    val tyFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    tyFactory.addRule(relativeDay, "relative days")
    tyFactory.findMatchFirstOnly(text).map (possibleDate => {
      val tyDate = possibleDate.content
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
    }})
  }

  /** Searches for exactly provided days of the week. Always relative from current time at processing */
  private def extractRelativeExactDay(text: String): Option[MatchedDateTime] = {
    val relativeExactFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    relativeExactFactory.addRule(relativeExactDay, "relative precise dates")
    relativeExactFactory.findMatchFirstOnly(text).map(possibleDate => {
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
    })
  }

  /**
    * Searches for times of the day
    * @param dateTime If any dates found previously, keep it as part of the final result
    * @param text target document
    * @return a final possible date if any found
    */
  private def setTimeIfAny(dateTime: Option[MatchedDateTime], text: String): Option[MatchedDateTime] = {
    val timeFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    timeFactory.addRule(clockTime, "standard time extraction")
    timeFactory.addRule(altTime, "alternative time format")
    timeFactory.addRule(coordTIme, "coordinate like time")
    timeFactory.addRule(refTime, "referred time")
    timeFactory.findMatchFirstOnly(text).map { possibleTime => {
      val calendarBuild = new Calendar.Builder
      val currentCalendar = dateTime.map(_.calendar).getOrElse(Calendar.getInstance)
      calendarBuild.setDate(
        currentCalendar.get(Calendar.YEAR),
        currentCalendar.get(Calendar.MONTH),
        currentCalendar.get(Calendar.DAY_OF_MONTH)
      )
      val times = possibleTime.content.subgroups
      val hour = {
        /** assuming PM if 2 digits regex-subgroup hour is defined, is ot AM and is less than number 12 e.g. meet you at 5*/
        if (
          times.head != null && // hour is defined
            amDefinition.findFirstIn(text).isDefined && // no explicit am
            times.head.toInt < 12 // hour is within smaller than 12
        ) times.head.toInt + 12
        else if (times.head.toInt < 25) times.head.toInt
        else 0
      }
      /** Minutes are valid if regex-subgroup matched and less than number 60*/
      val minutes = {
        if (times(1) != null && times(1).toInt < 60) times(1).toInt
        else 0
      }
      /** Seconds are valid if regex-subgroup matched and less than number 60*/
      val seconds = {
        if (times(2) != null && times(2).toInt < 60) times(2).toInt
        else 0
      }
      calendarBuild.setTimeOfDay(hour, minutes, seconds)
      MatchedDateTime(calendarBuild.build, possibleTime.content.start, possibleTime.content.end)
    }}
  }

  /** One to one relationship between content document and output annotation
    * @return Any found date, empty if not. Final format is [[dateFormat]] or default yyyy/MM/dd
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val simpleDateFormat = new SimpleDateFormat(getFormat)
    annotations.flatMap( annotation =>
      extractDate(annotation.result).map(matchedDate => Annotation(
        annotatorType,
        simpleDateFormat.format(matchedDate.calendar.getTime),
        Map(Annotation.BEGIN -> matchedDate.start.toString,
          Annotation.END -> (matchedDate.end - 1).toString)
      ))
    )
  }

}
object DateMatcher extends DefaultParamsReadable[DateMatcher]
