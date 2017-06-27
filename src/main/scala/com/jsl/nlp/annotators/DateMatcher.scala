package com.jsl.nlp.annotators

import java.text.SimpleDateFormat

import com.jsl.nlp.{Annotation, Annotator, Document}

import scala.util.matching.Regex
import java.util.Calendar

import com.jsl.nlp.util.regex.{MatchStrategy, RuleFactory}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by Saif Addin on 6/3/2017.
  */
class DateMatcher(override val uid: String) extends Annotator {

  private[annotators] case class MatchedDateTime(calendar: Calendar, start: Int, end: Int)

  /**
    * Formal date
    */
  private val formalDate = new Regex("(\\b\\d{2,4})[-/](\\d{1,2})[-/](\\d{1,2}\\b)", "year", "month", "day")
  private val formalDateAlt = new Regex("(\\b\\d{1,2})[-/](\\d{1,2})[-/](\\d{2,4}\\b)", "month", "day", "year")

  /**
    * Relaxed date
    */
  private val months = Seq("january","february","march","april","may","june","july","august","september","october","november","december")
  private val shortMonths = Seq("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec")

  private val relaxedDayNumbered = "\\b(\\d{1,2})(?:st|rd|nd|th)*\\b".r
  private val relaxedMonths = "(?i)" + months.mkString("|")
  private val relaxedShortMonths = "(?i)" + shortMonths.mkString("|")
  private val relaxedYear = "\\d{4}\\b|\\B'\\d{2}\\b".r

  /**
    * Relative dates
    */
  private val relativeDate = "(?i)(next|last)\\s(week|month|year)".r
  private val relativeDay = "(?i)(today|tomorrow|yesterday|past tomorrow|day before|day after|day before yesterday|day after tomorrow)".r
  private val relativeExactDay = "(?i)(next|last|past)\\s(mon|tue|wed|thu|fri)".r

  /**
    * time catch
    */
  private val clockTime = new Regex("(?i)([0-2][0-9]):([0-5][0-9])(?::([0-5][0-9]))?", "hour", "minutes", "seconds")
  private val altTime = new Regex("([0-2]?[0-9])\\.([0-5][0-9])\\.?([0-5][0-9])?", "hour", "minutes", "seconds")
  private val coordTIme = new Regex("([0-2]?[0-9])([0-5][0-9])?\\.?([0-5][0-9])?\\s*(?:h|a\\.?m|p\\.?m)", "hour", "minutes", "seconds")
  private val refTime = new Regex("at\\s+([0-9])\\s*([0-5][0-9])*\\s*([0-5][0-9])*")
  private val amDefinition = "(?i)(a\\.?m)".r

  protected val dateFormat: Param[String] = new Param(this, "Date Format", "SimpleDateFormat standard criteria")

  override val aType: String = DateMatcher.aType

  override var requiredAnnotationTypes: Array[String] = Array()

  def this() = this(Identifiable.randomUID(DateMatcher.aType))

  def getFormat: String = get(dateFormat).getOrElse("yyyy/MM/dd")

  def setFormat(value: String): this.type = set(dateFormat, value)

  private[annotators] def extractDate(text: String): Option[MatchedDateTime] = {
    val possibleDate = extractFormalDate(text)
      .orElse(extractRelaxedDate(text))
      .orElse(extractRelativeDate(text))
      .orElse(extractTomorrowYesterday(text))
      .orElse(extractRelativeExactDay(text))

    possibleDate.orElse(setTimeIfAny(possibleDate, text))
  }

  private def extractFormalDate(text: String): Option[MatchedDateTime] = {
    val formalFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    formalFactory.addRule(formalDate, "formal date matcher with year at first")
    formalFactory.addRule(formalDateAlt, "formal date with year at end")
    formalFactory.findMatchFirstOnly(text).map{ possibleDate =>
      val formalDate = possibleDate.content
      val calendar = new Calendar.Builder()
      MatchedDateTime(
        calendar.setDate(
          if (formalDate.group("year").toInt > 999) formalDate.group("year").toInt else formalDate.group("year").toInt + 1900,
          formalDate.group("month").toInt - 1,
          formalDate.group("day").toInt
        ).build(),
        formalDate.start,
        formalDate.end
      )
    }
  }

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

  private def extractTomorrowYesterday(text: String): Option[MatchedDateTime] = {
    val tyFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)
    tyFactory.addRule(relativeDay, "relative days")
    tyFactory.findMatchFirstOnly(text).map (possibleDate => {
      val tyDate = possibleDate.content
      tyDate.matched match {
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
        /**
          * we can assume time is pm if ->
          */
        if (
          times.head != null && // hour is defined
            amDefinition.findFirstIn(text).isDefined && // no explicit am
            times.head.toInt < 12 // hour is within smaller than 12
        ) times.head.toInt + 12
        else if (times.head.toInt < 25) times.head.toInt
        else 0
      }
      val minutes = {
        if (times(1) != null && times(1).toInt < 60) times(1).toInt
        else 0
      }
      val seconds = {
        if (times(2) != null && times(2).toInt < 60) times(2).toInt
        else 0
      }
      calendarBuild.setTimeOfDay(hour, minutes, seconds)
      MatchedDateTime(calendarBuild.build, possibleTime.content.start, possibleTime.content.end)
    }}
  }

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val simpleDateFormat = new SimpleDateFormat(getFormat)
    Seq(extractDate(document.text).map(matchedDate => Annotation(
      DateMatcher.aType,
      matchedDate.start,
      matchedDate.end,
      Map(DateMatcher.aType -> simpleDateFormat.format(matchedDate.calendar.getTime)))
    )).flatten
  }

}
object DateMatcher extends DefaultParamsReadable[DateMatcher] {
  val aType: String = "date"
}
