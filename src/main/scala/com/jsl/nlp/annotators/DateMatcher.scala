package com.jsl.nlp.annotators

import java.text.SimpleDateFormat

import com.jsl.nlp.{Annotation, Annotator, Document}

import scala.util.matching.Regex
import java.util.Calendar

import org.apache.spark.ml.param.Param

/**
  * Created by Saif Addin on 6/3/2017.
  */
class DateMatcher extends Annotator {

  private[annotators] case class MatchedDate(calendar: Calendar, start: Int, end: Int)

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

  protected val dateFormat: Param[SimpleDateFormat] = new Param(this, "Date Format", "SimpleDateFormat standard criteria")

  override val aType: String = DateMatcher.aType

  override val requiredAnnotationTypes: Array[String] = Array()

  def getFormat: String = get(dateFormat).map(_.toPattern).getOrElse("yyyy/MM/dd")

  private def getSDFormat: SimpleDateFormat = get(dateFormat).getOrElse(new SimpleDateFormat(getFormat))

  def setFormat(value: String): Unit = set(dateFormat, new SimpleDateFormat(value))

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    Seq(extractDate(document.text).map(matchedDate => Annotation(
      DateMatcher.aType,
      matchedDate.start,
      matchedDate.end,
      Map(DateMatcher.aType -> getSDFormat.format(matchedDate.calendar.getTime)))
    )).flatten
  }

  private[annotators] def extractDate(text: String): Option[MatchedDate] = {
    extractFormalDate(text)
      .orElse(extractRelaxedDate(text))
      .orElse(extractRelativeDate(text))
      .orElse(extractTomorrowYesterday(text))
      .orElse(extractRelativeExactDay(text))
  }

  private def extractFormalDate(text: String): Option[MatchedDate] = {
    formalDate.findFirstMatchIn(text).orElse(formalDateAlt.findFirstMatchIn(text)).map{m =>
      val calendar = new Calendar.Builder()
      MatchedDate(
        calendar.setDate(
          if (m.group("year").toInt > 999) m.group("year").toInt else m.group("year").toInt + 1900,
          m.group("month").toInt - 1,
          m.group("day").toInt
        ).build(),
        m.start,
        m.end
      )
    }
  }

  private def extractRelaxedDate(text: String): Option[MatchedDate] = {
    val calendar = new Calendar.Builder()
    val d = relaxedDayNumbered.findFirstMatchIn(text).map(m => (m.start, m.end, m.group(1).toInt))
    val m = relaxedMonths.r.findFirstMatchIn(text)
      .map(m => (m.start, m.end, months.indexOf(m.matched.toLowerCase)))
      .orElse(relaxedShortMonths.r.findFirstMatchIn(text)
      .map(m => (m.start, m.end, shortMonths.indexOf(m.matched.toLowerCase))))
    val y = relaxedYear.findFirstMatchIn(text).map(m => (m.start, m.end, m.matched.filter(_.isDigit).toInt))
    if (y.isDefined && m.isDefined && d.isDefined ) {
      println(y.get._3, m.get._3, d.get._3)
      calendar.setDate(if (y.get._3 > 999) y.get._3 else y.get._3 + 1900, m.get._3, d.get._3)
      Some(MatchedDate(
        calendar.build(),
        Seq(y.get._1, m.get._1, d.get._1).min,
        Seq(y.get._2, m.get._2, d.get._2).max
      ))
    } else if (m.isDefined && d.isDefined) {
      calendar.setDate(Calendar.getInstance().getWeekYear, m.get._3, d.get._3)
      Some(MatchedDate(
        calendar.build(),
        Seq(m.get._1, d.get._1).min,
        Seq(m.get._2, d.get._2).max
      ))
    } else None
  }

  private def extractRelativeDate(text: String): Option[MatchedDate] = {
    relativeDate.findFirstMatchIn(text).map(m => {
      val calendar = Calendar.getInstance()
      val amount = if (m.group(1) == "next") 1 else -1
      m.group(2) match {
        case "week" => calendar.add(Calendar.WEEK_OF_MONTH, amount)
        case "month" => calendar.add(Calendar.MONTH, amount)
        case "year" => calendar.add(Calendar.YEAR, amount)
      }
      MatchedDate(calendar, m.start, m.end)
    })
  }

  private def extractTomorrowYesterday(text: String): Option[MatchedDate] = {
    relativeDay.findFirstMatchIn(text).map (m => { m.matched match {
      case "today" =>
        val calendar = Calendar.getInstance()
        MatchedDate(calendar, m.start, m.end)
      case "tomorrow" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, 1)
        MatchedDate(calendar, m.start, m.end)
      case "past tomorrow" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, 2)
        MatchedDate(calendar, m.start, m.end)
      case "yesterday" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, -1)
        MatchedDate(calendar, m.start, m.end)
      case "day after" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, 1)
        MatchedDate(calendar, m.start, m.end)
      case "day before" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, -1)
        MatchedDate(calendar, m.start, m.end)
      case "day after tomorrow" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, 2)
        MatchedDate(calendar, m.start, m.end)
      case "day before yesterday" =>
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_MONTH, -2)
        MatchedDate(calendar, m.start, m.end)
    }})
  }

  private def extractRelativeExactDay(text: String): Option[MatchedDate] = {
    relativeExactDay.findFirstMatchIn(text).map(m => {
        val calendar = Calendar.getInstance()
        val amount = if (m.group(1) == "next") 1 else -1
        calendar.add(Calendar.DAY_OF_MONTH, amount)
        m.group(2) match {
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
      MatchedDate(calendar, m.start, m.end)
    })
  }
}
object DateMatcher {
  val aType: String = "date"
}
