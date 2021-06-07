package com.johnsnowlabs.nlp.annotators

import java.text.SimpleDateFormat
import java.util.Calendar

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Matches standard date formats into a provided format.
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
  * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/databricks_notebooks/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers_v3.0.ipynb Spark NLP Workshop]]
  * and the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/MultiDateMatcherTestSpec.scala MultiDateMatcherTestSpec]].
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
  * @param uid internal uid required to generate writable annotators
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
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class MultiDateMatcher(override val uid: String) extends AnnotatorModel[MultiDateMatcher] with HasSimpleAnnotate[MultiDateMatcher] with DateMatcherUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Output Annotator Type : DATE
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = DATE
  /** Input Annotator Type : DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Internal constructor to submit a random UID */
  def this() = this(Identifiable.randomUID("MULTI_DATE"))

  /**
    * Finds dates in a specific order, from formal to more relaxed. Add time of any, or stand-alone time
    *
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

  private def extractFormalDate(text: String): Seq[MatchedDateTime] = {
    formalFactory.findMatch(text).map{ possibleDate =>
      formalDateContentParse(possibleDate)
    }
  }

  private def extractRelaxedDate(text: String): Seq[MatchedDateTime] = {
    val possibleDates = relaxedFactory.findMatch(text)
    var dayMatch = $(defaultDayWhenMissing)
    var monthMatch = defaultMonthWhenMissing
    var yearMatch = defaultYearWhenMissing
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
