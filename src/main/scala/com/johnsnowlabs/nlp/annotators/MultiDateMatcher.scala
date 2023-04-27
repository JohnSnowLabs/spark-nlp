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
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.commons.lang3.time.DateUtils
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import java.text.SimpleDateFormat
import java.util.Calendar
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex

/** Matches standard date formats into a provided format.
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
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/MultiDateMatcherMultiLanguage_en.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/MultiDateMatcherTestSpec.scala MultiDateMatcherTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.MultiDateMatcher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val date = new MultiDateMatcher()
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
  * val data = Seq("I saw him yesterday and he told me that he will visit us next week")
  *   .toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(date) as dates").show(false)
  * +-----------------------------------------------+
  * |dates                                          |
  * +-----------------------------------------------+
  * |[date, 57, 65, 2020/01/18, [sentence -> 0], []]|
  * |[date, 10, 18, 2020/01/10, [sentence -> 0], []]|
  * +-----------------------------------------------+
  * }}}
  *
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
class MultiDateMatcher(override val uid: String)
    extends AnnotatorModel[MultiDateMatcher]
    with HasSimpleAnnotate[MultiDateMatcher]
    with DateMatcherUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Output Annotator Type : DATE
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DATE

  /** Input Annotator Type : DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Internal constructor to submit a random UID */
  def this() = this(Identifiable.randomUID("MULTI_DATE"))

  private def runTranslation(text: String) = {
    val sourceLanguage = getSourceLanguage
    val translationPreds = Array(sourceLanguage.length == 2, !sourceLanguage.equals("en"))

    if (translationPreds.forall(_.equals(true)))
      new DateMatcherTranslator(MultiDatePolicy).translate(text, sourceLanguage)
    else
      text
  }

  private def findByInputFormatsRules(text: String, factory: RuleFactory): Seq[MatchedDateTime] =
    factory
      .findMatch(text)
      .map(formalDateContentParse(_))
      .groupBy(_.calendar)
      .map { case (_, group) => group.head }
      .toSeq

  def runInputFormatsSearch(text: String): Seq[MatchedDateTime] = {
    val regexes: Array[Regex] = getInputFormats
      .filter(formalInputFormats.contains(_))
      .map(formalInputFormats(_))

    for (r <- regexes) {
      formalFactoryInputFormats.addRule(r, "formal rule from input formats")
    }

    findByInputFormatsRules(text, formalFactoryInputFormats)
  }

  def runDateExtractorChain(_text: String): Seq[MatchedDateTime] = {
    val strategies: Seq[() => Seq[MatchedDateTime]] = Seq(
      () => extractFormalDate(_text),
      () => extractRelativeDatePast(_text),
      () => extractRelativeDateFuture(_text),
      () => extractRelaxedDate(_text),
      () => extractRelativeDate(_text),
      () => extractTomorrowYesterday(_text),
      () => extractRelativeExactDay(_text))

    strategies.foldLeft(Seq.empty[MatchedDateTime])((previousResults, strategy) => {
      // Always keep earliest match of each strategy by date found
      val newResults = strategy()
      newResults.foldLeft(previousResults)((previous, newResult) => {
        // Prioritize previous results on this index, ignore new ones if overlapping previous results
        if (previous.exists(_.start == newResult.start))
          previous
        else
          previous :+ newResult
      })
    })
  }

  /** Finds dates in a specific order, from formal to more relaxed. Add time of any, or
    * stand-alone time
    *
    * @param text
    *   input text coming from target document
    * @return
    *   a possible date-time match
    */
  private[annotators] def extractDate(text: String): Seq[MatchedDateTime] = {

    val _text: String = runTranslation(text)

    def inputFormatsAreDefined = !getInputFormats.sameElements(EMPTY_INIT_ARRAY)

    val possibleDates: Seq[MatchedDateTime] =
      if (inputFormatsAreDefined)
        runInputFormatsSearch(_text)
      else
        runDateExtractorChain(_text)

    possibleDates
  }

  private def extractRelativeDateFuture(text: String): Seq[MatchedDateTime] = {
    if ("(.*)\\s*in\\s*[0-9](.*)".r.findFirstMatchIn(text).isDefined)
      relativeFutureFactory
        .findMatch(text.toLowerCase())
        .map(possibleDate => relativeDateFutureContentParse(possibleDate))
    else
      Seq.empty
  }

  private def extractRelativeDatePast(text: String): Seq[MatchedDateTime] = {
    if ("(.*)\\s*[0-9]\\s*(.*)\\s*(ago)(.*)".r.findFirstMatchIn(text).isDefined)
      relativePastFactory
        .findMatch(text.toLowerCase())
        .map(possibleDate => relativeDatePastContentParse(possibleDate))
    else
      Seq.empty
  }

  private def extractFormalDate(text: String): Seq[MatchedDateTime] = {
    val allFormalDateMatches = formalFactory.findMatch(text).map { possibleDate =>
      formalDateContentParse(possibleDate)
    }

    regularizeFormalDateMatches(allFormalDateMatches)
  }

  private def regularizeFormalDateMatches: Seq[MatchedDateTime] => Seq[MatchedDateTime] =
    allFormalDateMatches => {
      def truncatedExists(e: Calendar, candidate: Calendar) = {
        DateUtils.truncate(e, Calendar.MONTH).equals(candidate)
      }

      val indexedMatches: Seq[(MatchedDateTime, Int)] = allFormalDateMatches.zipWithIndex
      val indexesToRemove = new ListBuffer[Int]()

      for (e <- indexedMatches) {
        val candidates = indexedMatches.filterNot(_._2 == e._2)
        val accTempIdx: Seq[Int] =
          for (candidate <- candidates
            // if true, the candidate is the truncated match of the existing match
            if truncatedExists(e._1.calendar, candidate._1.calendar)) yield candidate._2
        accTempIdx.foreach(indexesToRemove.append(_))
      }

      val regularized =
        indexedMatches.filterNot { case (_, i) => indexesToRemove.contains(i) }.map(_._1)
      regularized
    }

  private def extractRelaxedDate(text: String): Seq[MatchedDateTime] = {
    val possibleDates = relaxedFactory.findMatch(text)
    var dayMatch = $(defaultDayWhenMissing)
    var monthMatch = defaultMonthWhenMissing
    var yearMatch = defaultYearWhenMissing
    var changes = 0

    possibleDates.foreach(possibleDate => {

      if (possibleDate.identifier == "relaxed days" && possibleDate.content.matched.exists(
          _.isDigit)) {
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
      Seq(
        MatchedDateTime(
          calendar.build(),
          possibleDates.map(_.content.start).min,
          possibleDates.map(_.content.end).max))
    } else Seq.empty
  }

  private def extractRelativeDate(text: String): Seq[MatchedDateTime] = {
    relativeFactory.findMatch(text).map(possibleDate => relativeDateContentParse(possibleDate))
  }

  private def extractTomorrowYesterday(text: String): Seq[MatchedDateTime] = {
    tyFactory
      .findMatch(text)
      .map(possibleDate => tomorrowYesterdayContentParse(possibleDate))
  }

  private def extractRelativeExactDay(text: String): Seq[MatchedDateTime] = {
    relativeExactFactory
      .findMatch(text.toLowerCase)
      .map(possibleDate => relativeExactContentParse(possibleDate))
  }

  /** One to one relationship between content document and output annotation
    *
    * @return
    *   Any found date, empty if not. Final format is [[outputFormat]] or default yyyy/MM/dd
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val simpleDateFormat = new SimpleDateFormat(getOutputFormat)
    annotations.flatMap(annotation =>
      extractDate(annotation.result)
        .map(matchedDate =>
          Annotation(
            outputAnnotatorType,
            matchedDate.start,
            matchedDate.end - 1,
            simpleDateFormat.format(matchedDate.calendar.getTime),
            annotation.metadata)))
  }

}

/** This is the companion object of [[MultiDateMatcher]]. Please refer to that class for the
  * documentation.
  */
object MultiDateMatcher extends DefaultParamsReadable[MultiDateMatcher]
