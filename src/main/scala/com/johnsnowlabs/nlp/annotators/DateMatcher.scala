package com.johnsnowlabs.nlp.annotators

import java.text.SimpleDateFormat

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}

import java.util.Calendar

import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Matches standard date formats into a provided format
  * @param uid internal uid required to generate writable annotators
  * @@ dateFormat: allows to define expected output format. Follows SimpleDateFormat standard.
  */
class DateMatcher(override val uid: String) extends AnnotatorModel[DateMatcher] with DateMatcherUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val outputAnnotatorType: AnnotatorType = DATE

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  setDefault(
    inputCols -> Array(DOCUMENT),
    dateFormat -> "yyyy/MM/dd"
  )

  /** Internal constructor to submit a random UID */
  def this() = this(Identifiable.randomUID("DATE"))

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


  private def extractFormalDate(text: String): Option[MatchedDateTime] = {
    formalFactory.findMatchFirstOnly(text).map{ possibleDate =>
      formalDateContentParse(possibleDate)
    }
  }

  private def extractRelaxedDate(text: String): Option[MatchedDateTime] = {
    val possibleDates = relaxedFactory.findMatch(text)
    if (possibleDates.length > 1) {
      var dayMatch = 1
      var monthMatch = 1
      var yearMatch = Calendar.getInstance().getWeekYear

      val dayCandidate = possibleDates.find(_.identifier == "relaxed days")
      if (dayCandidate.isDefined && dayCandidate.get.content.matched.exists(_.isDigit)) {
        dayMatch = dayCandidate.get.content.matched.filter(_.isDigit).toInt
      }

      val monthCandidate = possibleDates.find(_.identifier == "relaxed months exclusive")
      if (monthCandidate.isDefined && monthCandidate.get.content.matched.length > 2) {
        val month = monthCandidate.get.content.matched.toLowerCase().take(3)
        if (shortMonths.contains(month))
          monthMatch = shortMonths.indexOf(month)
      }

      val yearCandidate = possibleDates.find(_.identifier == "relaxed year")
      if (yearCandidate.isDefined &&
        yearCandidate.get.content.matched.exists(_.isDigit) &&
        yearCandidate.get.content.matched.length > 2) {
        val year = yearCandidate.get.content.matched.filter(_.isDigit).toInt
        yearMatch = if (year > 999) year else year + 1900
      }

      val calendar = new Calendar.Builder()
      calendar.setDate(yearMatch, monthMatch, dayMatch)
      val matches = possibleDates.map(p => (p.content.start, p.content.end))
      Some(MatchedDateTime(
        calendar.build(),
        matches.minBy(_._1)._1,
        matches.maxBy(_._2)._2
      ))
    } else None
  }

  private def extractRelativeDate(text: String): Option[MatchedDateTime] = {
    relativeFactory.findMatchFirstOnly(text).map(possibleDate =>
      relativeDateContentParse(possibleDate)
    )
  }

  private def extractTomorrowYesterday(text: String): Option[MatchedDateTime] = {
    tyFactory.findMatchFirstOnly(text).map (possibleDate =>
      tomorrowYesterdayContentParse(possibleDate)
    )
  }

  private def extractRelativeExactDay(text: String): Option[MatchedDateTime] = {
    relativeExactFactory.findMatchFirstOnly(text).map(possibleDate =>
      relativeExactContentParse(possibleDate)
    )
  }

  private def setTimeIfAny(dateTime: Option[MatchedDateTime], text: String): Option[MatchedDateTime] = {
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
        outputAnnotatorType,
        matchedDate.start,
        matchedDate.end - 1,
        simpleDateFormat.format(matchedDate.calendar.getTime),
        annotation.metadata
      ))
    )
  }

}
object DateMatcher extends DefaultParamsReadable[DateMatcher]
