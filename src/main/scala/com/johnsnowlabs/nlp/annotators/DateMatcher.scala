/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.util.regex.RuleFactory
import com.johnsnowlabs.nlp.util.regex.RuleFactory.RuleMatch
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import java.text.SimpleDateFormat
import java.util.Calendar
import scala.util.matching.Regex

/** Matches standard date formats into a provided format Reads from different forms of date and
  * time expressions and converts them to a provided date format.
  *
  * Extracts only '''one''' date per document. Use with sentence detector to find matches in each
  * sentence. To extract multiple dates from a document, please use the [[MultiDateMatcher]].
  *
  * Reads the following kind of dates:
  * {{{
  * "1978-01-28", "1984/04/02,1/02/1980", "2/28/79", "The 31st of April in the year 2008",
  * "Fri, 21 Nov 1997", "Jan 21, ‘97", "Sun", "Nov 21", "jan 1st", "next thursday",
  * "last wednesday", "today", "tomorrow", "yesterday", "next week", "next month",
  * "next year", "day after", "the day before", "0600h", "06:00 hours", "6pm", "5:30 a.m.",
  * "at 5", "12:59", "23:59", "1988/11/23 6pm", "next week at 7.30", "5 am tomorrow"
  * }}}
  *
  * For example `"The 31st of April in the year 2008"` will be converted into `2008/04/31`.
  *
  * Pretrained pipelines are available for this module, see
  * [[https://nlp.johnsnowlabs.com/docs/en/pipelines Pipelines]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/annotation/text/english/MultiDateMatcherMultiLanguage_en.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DateMatcherTestSpec.scala DateMatcherTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.DateMatcher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val date = new DateMatcher()
  *   .setInputCols("document")
  *   .setOutputCol("date")
  *   .setAnchorDateYear(2020)
  *   .setAnchorDateMonth(1)
  *   .setAnchorDateDay(11)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   date
  * ))
  *
  * val data = Seq("Fri, 21 Nov 1997", "next week at 7.30", "see you a day after").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("date").show(false)
  * +-------------------------------------------------+
  * |date                                             |
  * +-------------------------------------------------+
  * |[[date, 5, 15, 1997/11/21, [sentence -> 0], []]] |
  * |[[date, 0, 8, 2020/01/18, [sentence -> 0], []]]  |
  * |[[date, 10, 18, 2020/01/12, [sentence -> 0], []]]|
  * +-------------------------------------------------+
  * }}}
  *
  * @see
  *   [[MultiDateMatcher]] for matching multiple dates in a document
  * @param uid
  *   internal uid required to generate writable annotators
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class DateMatcher(override val uid: String)
    extends AnnotatorModel[DateMatcher]
    with HasSimpleAnnotate[DateMatcher]
    with DateMatcherUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Output annotator type: DATE
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DATE

  /** Input annotator type: DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Internal constructor to submit a random UID */
  def this() = this(Identifiable.randomUID("DATE"))

  private def runFormalFactoryForInputFormats(
      text: String,
      factory: RuleFactory): Option[MatchedDateTime] = {
    factory.findMatchFirstOnly(text).map { possibleDate =>
      formalDateContentParse(possibleDate)
    }
  }

  def runInputFormatsSearch(text: String): Option[MatchedDateTime] = {
    val regexes: Array[Regex] = getInputFormats
      .filter(formalInputFormats.contains(_))
      .map(formalInputFormats(_))

    for (r <- regexes) {
      formalFactoryInputFormats.addRule(r, "formal rule from input formats")
    }

    runFormalFactoryForInputFormats(text, formalFactoryInputFormats)
  }

  /** Finds dates in a specific order, from formal to more relaxed. Add time of any, or
    * stand-alone time
    *
    * @param text
    *   input text coming from target document
    * @return
    *   a possible date-time match
    */
  private[annotators] def extractDate(text: String): Option[MatchedDateTime] = {

    val _text: String = runTranslation(text)

    def inputFormatsAreDefined = !getInputFormats.sameElements(EMPTY_INIT_ARRAY)

    val possibleDate: Option[MatchedDateTime] =
      if (inputFormatsAreDefined)
        runInputFormatsSearch(_text)
      else
        runDateExtractorChain(_text)

    possibleDate.orElse(setTimeIfAny(possibleDate, _text))
  }

  private def runDateExtractorChain(_text: String) = {
    extractFormalDate(_text)
      .orElse(extractRelativeDatePast(_text))
      .orElse(extractRelativeDateFuture(_text))
      .orElse(extractRelaxedDate(_text))
      .orElse(extractRelativeDate(_text))
      .orElse(extractTomorrowYesterday(_text))
      .orElse(extractRelativeExactDay(_text))
  }

  private def runTranslation(text: String) = {
    val sourceLanguage = getSourceLanguage
    val translationPreds = Array(sourceLanguage.length == 2, !sourceLanguage.equals("en"))

    if (translationPreds.forall(_.equals(true)))
      new DateMatcherTranslator(SingleDatePolicy).translate(text, sourceLanguage)
    else
      text
  }

  private def extractFormalDate(text: String): Option[MatchedDateTime] = {
    formalFactory.findMatchFirstOnly(text).map { possibleDate =>
      formalDateContentParse(possibleDate)
    }
  }

  private def isNotMonthSubwordMatch(text: String, d: RuleMatch): Boolean = {
    val words = text.replaceAll("""([?.!:]|\b\p{IsLetter}{1,2}\b)\s*""", "").split(SPACE_CHAR)
    val notSubWordMatches = words
      .map(_.toLowerCase)
      .filter(w =>
        w.contains(d.content.matched.toLowerCase) && w.length <= d.content.matched.length)

    notSubWordMatches.length match {
      case 1 => true
      case _ => false
    }
  }

  private def extractRelaxedDate(text: String): Option[MatchedDateTime] = {
    val possibleDates: Seq[RuleFactory.RuleMatch] = relaxedFactory.findMatch(text)

    if (possibleDates.length > 1) {
      var dayMatch = $(defaultDayWhenMissing)
      var monthMatch = defaultMonthWhenMissing
      var yearMatch = defaultYearWhenMissing

      val dayCandidate = possibleDates.find(_.identifier == "relaxed days")
      if (dayCandidate.isDefined && dayCandidate.get.content.matched.exists(_.isDigit)) {
        dayMatch = dayCandidate.get.content.matched.filter(_.isDigit).toInt
      }

      val monthCandidate = possibleDates
        .find(_.identifier == "relaxed months exclusive")
        .filter(d => isNotMonthSubwordMatch(text, d))

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
      Some(MatchedDateTime(calendar.build(), matches.minBy(_._1)._1, matches.maxBy(_._2)._2))
    } else None
  }

  private def extractRelativeDateFuture(text: String): Option[MatchedDateTime] = {
    if ("in\\s[0-9]".r.findFirstMatchIn(text).isDefined && !text.contains(relativePastPattern))
      relativeFutureFactory
        .findMatchFirstOnly(text.toLowerCase())
        .map(possibleDate => relativeDateFutureContentParse(possibleDate))
    else
      None
  }

  private def extractRelativeDatePast(text: String): Option[MatchedDateTime] = {
    if (!"(.*)\\s+(in)\\s+[0-9]".r.findFirstMatchIn(text).isDefined && text.contains(
        relativePastPattern))
      relativePastFactory
        .findMatchFirstOnly(text.toLowerCase())
        .map(possibleDate => relativeDatePastContentParse(possibleDate))
    else
      None
  }

  private def extractRelativeDate(text: String): Option[MatchedDateTime] = {
    if (!"in\\s+[0-9]".r.findFirstMatchIn(text).isDefined && !text.contains(relativePastPattern))
      relativeFactory
        .findMatchFirstOnly(text.toLowerCase)
        .map(possibleDate => relativeDateContentParse(possibleDate))
    else
      None
  }

  private def extractTomorrowYesterday(text: String): Option[MatchedDateTime] = {
    tyFactory
      .findMatchFirstOnly(text.toLowerCase())
      .map(possibleDate => tomorrowYesterdayContentParse(possibleDate))
  }

  private def extractRelativeExactDay(text: String): Option[MatchedDateTime] = {
    relativeExactFactory
      .findMatchFirstOnly(text.toLowerCase())
      .map(possibleDate => relativeExactContentParse(possibleDate))
  }

  private def setTimeIfAny(
      dateTime: Option[MatchedDateTime],
      text: String): Option[MatchedDateTime] = {
    timeFactory.findMatchFirstOnly(text).map { possibleTime =>
      {
        val calendarBuild = new Calendar.Builder
        val currentCalendar = dateTime.map(_.calendar).getOrElse(Calendar.getInstance)
        calendarBuild.setDate(
          currentCalendar.get(Calendar.YEAR),
          currentCalendar.get(Calendar.MONTH),
          currentCalendar.get(Calendar.DAY_OF_MONTH))
        val times = possibleTime.content.subgroups
        val hour = {

          /** assuming PM if 2 digits regex-subgroup hour is defined, is ot AM and is less than
            * number 12 e.g. meet you at 5
            */
          if (times.head != null && // hour is defined
            amDefinition.findFirstIn(text).isDefined && // no explicit am
            times.head.toInt < 12 // hour is within smaller than 12
          ) times.head.toInt + 12
          else if (times.head.toInt < 25) times.head.toInt
          else 0
        }

        /** Minutes are valid if regex-subgroup matched and less than number 60 */
        val minutes = {
          if (times(1) != null && times(1).toInt < 60) times(1).toInt
          else 0
        }

        /** Seconds are valid if regex-subgroup matched and less than number 60 */
        val seconds = {
          if (times(2) != null && times(2).toInt < 60) times(2).toInt
          else 0
        }
        calendarBuild.setTimeOfDay(hour, minutes, seconds)
        MatchedDateTime(calendarBuild.build, possibleTime.content.start, possibleTime.content.end)
      }
    }
  }

  /** One to one relationship between content document and output annotation
    *
    * @return
    *   Any found date, empty if not. Final format is [[outputFormat]] or default yyyy/MM/dd
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val simpleDateFormat = new SimpleDateFormat(getOutputFormat)
    annotations.flatMap(annotation =>
      extractDate(annotation.result).map(matchedDate =>
        Annotation(
          outputAnnotatorType,
          matchedDate.start,
          matchedDate.end - 1,
          simpleDateFormat.format(matchedDate.calendar.getTime),
          annotation.metadata)))
  }

}

/** This is the companion object of [[DateMatcher]]. Please refer to that class for the
  * documentation.
  */
object DateMatcher extends DefaultParamsReadable[DateMatcher]
