package com.jsl.nlp.annotators

import java.text.SimpleDateFormat

import com.jsl.nlp.{Annotation, Annotator, Document}
import java.util.Calendar

import org.apache.spark.ml.param.Param

/**
  * Created by Saif Addin on 6/3/2017.
  */
class DateMatcher extends Annotator {

  case class MatchedDate(calendar: Calendar, start: Int, end: Int)

  /**
    * Formal date
    */
  private val formalDate = "(\\b\\d{1,4})[-/](\\d{2})[-/](\\d{4}\\b|\\d{2}\\b)".r

  /**
    * Relaxed date
    */
  private val months = Seq("january","february","march","april","may","june","july","august","september","october","november","december")
  private val shortMonths = Seq("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec")

  private val relaxedDayNumbered = "\\b\\d{1,2}(?=>st|rd|nd|th|\\b)".r
  private val relaxedMonths = ("(?i)" +: months).mkString("|")
  private val relaxedShortMonths = ("(?i)" +: shortMonths).mkString("|")
  private val relaxedYear = "\\d{4}\\b|\\d{2}\\b".r

  /**
    * Relative dates
    */
  private val relativeDate = "(?i)(next|last)\\s(week|month|year)".r
  private val relativeDay = "(?i)(tomorrow|yesterday|past tomorrow|day before yesterday|day after tomorrow)".r
  private val relativeExactDay = "(?i)(next|last|past)\\s(mon|tue|wed|thu|fri)".r

  private val dateFormat: Param[SimpleDateFormat] = new Param(this, "Date Format", "SimpleDateFormat standard criteria")

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

  private def extractDate(text: String): Option[MatchedDate] = {
    extractFormalDate(text)
      .orElse(extractRelaxedDate(text))
      .orElse(extractRelativeDate(text))
      .orElse(extractTomorrowYesterday(text))
      .orElse(extractRelativeExactDay(text))
  }

  private def extractFormalDate(text: String): Option[MatchedDate] = {
    formalDate.findFirstMatchIn(text).map {m =>
      val calendar = new Calendar.Builder()
      MatchedDate(
        calendar.setDate(m.group(1).toInt, m.group(2).toInt, m.group(3).toInt).build(),
        m.start,
        m.end
      )
    }
  }

  private def extractRelaxedDate(text: String): Option[MatchedDate] = {
    val calendar = new Calendar.Builder()
    val d = relaxedDayNumbered.findFirstMatchIn(text).map(m => (m.start, m.end, m.matched.toInt))
    val m = relaxedMonths.r.findFirstMatchIn(text)
      .orElse(relaxedShortMonths.r.findFirstMatchIn(text))
      .map(m => (m.start, m.end, relaxedMonths.indexOf(m.matched) + 1))
    val y = relaxedYear.findFirstMatchIn(text).map(m => (m.start, m.end, m.matched.toInt))
    if (y.isDefined && m.isDefined && d.isDefined ) {
      calendar.setDate(y.get._3, m.get._3, d.get._3)
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
        calendar.add(Calendar.WEEK_OF_MONTH, amount)
        m.group(2) match {
          case "monday" =>
            while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.MONDAY) {
              calendar.add(Calendar.DAY_OF_MONTH, 1)
            }
          case "tuesday" =>
            while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.TUESDAY) {
              calendar.add(Calendar.DAY_OF_MONTH, 1)
            }
          case "wednesday" =>
            while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.WEDNESDAY) {
              calendar.add(Calendar.DAY_OF_MONTH, 1)
            }
          case "thursday" =>
            while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.THURSDAY) {
              calendar.add(Calendar.DAY_OF_MONTH, 1)
            }
          case "friday" =>
            while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.FRIDAY) {
              calendar.add(Calendar.DAY_OF_MONTH, 1)
            }
        }
      MatchedDate(calendar, m.start, m.end)
    })
  }
}
object DateMatcher {
  val aType: String = "date"
}
