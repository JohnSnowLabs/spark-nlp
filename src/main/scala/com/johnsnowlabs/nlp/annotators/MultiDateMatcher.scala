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
class MultiDateMatcher(override val uid: String) extends AnnotatorModel[MultiDateMatcher] with DateMatcherUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val outputAnnotatorType: AnnotatorType = DATE

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  setDefault(
    inputCols -> Array(DOCUMENT),
    dateFormat -> "yyyy/MM/dd"
  )

  /** Internal constructor to submit a random UID */
  def this() = this(Identifiable.randomUID("MULTI_DATE"))

  /**
    * Finds dates in a specific order, from formal to more relaxed. Add time of any, or stand-alone time
    * @param text input text coming from target document
    * @return a possible date-time match
    */
  private[annotators] def extractDate(text: String): Seq[MatchedDateTime] = {
    val strategies = Seq(
      () => extractFormalDate(text),
      () => extractRelaxedDate(text),
      () => extractRelativeDate(text),
      () => extractTomorrowYesterday(text),
      () => extractRelativeExactDay(text)
    )

    strategies.foldLeft(Seq.empty[MatchedDateTime])((previousResults, strategy) => {
      // Always keep earliest match of each strategy by date found
      val newResults = strategy().zipWithIndex.groupBy(_._1.start).map{case (_, dates) => dates.minBy(_._2)._1}.toSeq
      newResults.foldLeft(previousResults)((previous, newResult) => {
        // Prioritize previous results on this index, ignore new ones if overlapping previous results
        if (previous.exists(_.start == newResult.start))
          previous
        else
          previous :+ newResult
      })
    })

  }

  private def extractFormalDate(text: String): Seq[MatchedDateTime] = {
    formalFactory.findMatch(text).map{ possibleDate =>
      formalDateContentParse(possibleDate)
    }
  }

  private def extractRelaxedDate(text: String): Seq[MatchedDateTime] = {
    val possibleDates = relaxedFactory.findMatch(text)
    var dayMatch = 1
    var monthMatch = 1
    var yearMatch = Calendar.getInstance().getWeekYear
    var changes = 0
    possibleDates.foreach(possibleDate => {

      if (possibleDate.identifier == "relaxed days" && possibleDate.content.matched.exists(_.isDigit)) {
        changes += 1
        dayMatch = possibleDate.content.matched.filter(_.isDigit).toInt
      }

      if (possibleDate.identifier == "relaxed months exclusive" && possibleDate.content.matched.length > 2) {
        changes += 1
        val month = possibleDate.content.matched.toLowerCase().take(3)
        if (shortMonths.contains(month))
          monthMatch = shortMonths.indexOf(month)
      }

      if (possibleDate.identifier == "relaxed year" &&
        possibleDate.content.matched.exists(_.isDigit) &&
        possibleDate.content.matched.length > 2) {
        changes += 1
        val year = possibleDate.content.matched.filter(_.isDigit).toInt
        yearMatch = if (year > 999) year else year + 1900
      }
    })
    if (possibleDates.nonEmpty && changes > 1) {
      val calendar = new Calendar.Builder()
      calendar.setDate(yearMatch, monthMatch, dayMatch)
      Seq(MatchedDateTime(
        calendar.build(),
        possibleDates.map(_.content.start).min,
        possibleDates.map(_.content.end).max
      ))
    } else Seq.empty
  }

  private def extractRelativeDate(text: String): Seq[MatchedDateTime] = {
    relativeFactory.findMatch(text).map(possibleDate =>
      relativeDateContentParse(possibleDate)
    )
  }

  private def extractTomorrowYesterday(text: String): Seq[MatchedDateTime] = {
    tyFactory.findMatch(text).map (possibleDate =>
      tomorrowYesterdayContentParse(possibleDate)
    )
  }

  private def extractRelativeExactDay(text: String): Seq[MatchedDateTime] = {
    relativeExactFactory.findMatch(text).map(possibleDate =>
      relativeExactContentParse(possibleDate)
    )
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
object MultiDateMatcher extends DefaultParamsReadable[MultiDateMatcher]
