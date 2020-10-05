package com.johnsnowlabs.nlp.annotators

import java.text.SimpleDateFormat
import java.util.Calendar

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Matches standard date formats into a provided format
  * Reads from different forms of date and time expressions and converts them to a provided date format. Extracts only ONE date per sentence. Use with sentence detector for more matches.
  *
  * Reads the following kind of dates:
  *
  * 1978-01-28, 1984/04/02,1/02/1980, 2/28/79, The 31st of April in the year 2008, "Fri, 21 Nov 1997" , "Jan 21, ‘97" , Sun, Nov 21, jan 1st, next thursday, last wednesday, today, tomorrow, yesterday, next week, next month, next year, day after, the day before, 0600h, 06:00 hours, 6pm, 5:30 a.m., at 5, 12:59, 23:59, 1988/11/23 6pm, next week at 7.30, 5 am tomorrow
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DateMatcherTestSpec.scala]] for further reference on how to use this API
  *
  * @param uid internal uid required to generate writable annotators
  * @@ dateFormat: allows to define expected output format. Follows SimpleDateFormat standard.
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class DateMatcher(override val uid: String) extends AnnotatorModel[DateMatcher] with HasSimpleAnnotate[DateMatcher] with DateMatcherUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Output annotator type: DATE
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = DATE

  /** Input annotator type: DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Internal constructor to submit a random UID */
  def this() = this(Identifiable.randomUID("DATE"))

  /**
    * Finds dates in a specific order, from formal to more relaxed. Add time of any, or stand-alone time
    *
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
      var dayMatch = $(defaultDayWhenMissing)
      var monthMatch = defaultMonthWhenMissing
      var yearMatch = defaultYearWhenMissing

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
