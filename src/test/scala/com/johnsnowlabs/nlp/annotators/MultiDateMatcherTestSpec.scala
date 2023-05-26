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

import com.johnsnowlabs.nlp.AnnotatorType.DATE
import com.johnsnowlabs.nlp.util.io.MatchStrategy
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DataBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

import java.util.Calendar

class MultiDateMatcherTestSpec extends AnyFlatSpec with DateMatcherBehaviors {

  val dateMatcher = new MultiDateMatcher
  "a MultiDateMatcher" should s"be of type ${AnnotatorType.DATE}" taggedAs FastTest in {
    assert(dateMatcher.outputAnnotatorType == AnnotatorType.DATE)
  }

  val dateData: Dataset[Row] =
    DataBuilder.multipleDataBuild(Array("2014/01/23", "day after tomorrow"))

  "A full MultiDateMatcher pipeline with some sentences" should behave like sparkBasedDateMatcher(
    dateData)

  val currentYear: Int = Calendar.getInstance.get(Calendar.YEAR)
  val nextThursdayCalendar: Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, 1)
    while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.THURSDAY)
      calendar.add(Calendar.DAY_OF_MONTH, 1)
    calendar
  }
  val lastWednesdayCalendar: Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, -1)
    while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.WEDNESDAY)
      calendar.add(Calendar.DAY_OF_MONTH, -1)
    calendar
  }
  val tomorrowCalendar: Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, 1)
    calendar
  }
  val yesterdayCalendar: Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, -1)
    calendar
  }

  def nextCalendar(which: Int): Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(which, 1)
    calendar
  }

  def setTimeTo(calendar: Calendar, hour: Int, minutes: Int, seconds: Int): Calendar = {
    val calendarBuild = new Calendar.Builder
    calendarBuild.setDate(
      calendar.get(Calendar.YEAR),
      calendar.get(Calendar.MONTH),
      calendar.get(Calendar.DAY_OF_MONTH))
    calendarBuild.setTimeOfDay(hour, minutes, seconds)
    calendarBuild.build
  }

  val dateSentences: Array[(String, Option[Calendar])] = Array(
    ("1978-01-28", Some(new Calendar.Builder().setDate(1978, 1 - 1, 28).build)),
    ("1984/04/02", Some(new Calendar.Builder().setDate(1984, 4 - 1, 2).build)),
    ("1/02/1980", Some(new Calendar.Builder().setDate(1980, 1 - 1, 2).build)),
    ("2/28/79", Some(new Calendar.Builder().setDate(1979, 2 - 1, 28).build)),
    (
      "The 31st of April in the year 2008",
      Some(new Calendar.Builder().setDate(2008, 4 - 1, 31).build)),
    ("Fri, 21 Nov 1997", Some(new Calendar.Builder().setDate(1997, 11 - 1, 21).build)),
    ("Jan 21, '97", Some(new Calendar.Builder().setDate(1997, 1 - 1, 21).build)),
    ("Sun, Nov 21", Some(new Calendar.Builder().setDate(currentYear, 11 - 1, 21).build)),
    ("jan 1st", Some(new Calendar.Builder().setDate(currentYear, 1 - 1, 1).build)),
    // NS: "february twenty-eighth",
    ("next thursday", Some(nextThursdayCalendar)),
    ("last wednesday", Some(lastWednesdayCalendar)),
    ("today", Some(Calendar.getInstance)),
    ("tomorrow", Some(tomorrowCalendar)),
    ("yesterday", Some(yesterdayCalendar)),
    ("next week", Some(nextCalendar(Calendar.WEEK_OF_MONTH))),
    ("next month", Some(nextCalendar(Calendar.MONTH))),
    ("next year", Some(nextCalendar(Calendar.YEAR))),
    // NS: "3 days from now",
    // NS: "three weeks ago",
    (" day after", Some(tomorrowCalendar)),
    ("the day before", Some(yesterdayCalendar)),
    // "the monday after",
    // "the monday before"
    // NS: "2 fridays before",
    // NS: "4 tuesdays after"
    (
      "Let's meet on 20th of February.",
      Some(new Calendar.Builder().setDate(currentYear, 2 - 1, 20).build)),
    ("Today is March 14th 2019.", Some(new Calendar.Builder().setDate(2019, 3 - 1, 14).build)),
    ("10-02-19", Some(new Calendar.Builder().setDate(2019, 10 - 1, 2).build)),
    // Breaking use cases
    ("June 2015", Some(new Calendar.Builder().setDate(2015, 6 - 1, 1).build)),
    ("August 2016", Some(new Calendar.Builder().setDate(2016, 8 - 1, 1).build)),
    ("4", None),
    ("L-2", None),
    ("Tarceva", None),
    ("2", None),
    ("3", None),
    ("Xgeva", None),
    ("today 4", Some(Calendar.getInstance)),
    ("1 month", None))

  dateSentences
    .map(date => dateMatcher.extractDate(date._1))
    .zip(dateSentences)
    .foreach(dateAnswer => {
      "a MultiDateMatcher" should s"successfully parse ${dateAnswer._2._1} as ${dateAnswer._2._2
          .map(_.getTime)}" taggedAs FastTest in {
        if (dateAnswer._1.isEmpty && dateAnswer._2._2.isEmpty)
          succeed
        else if (dateAnswer._1.nonEmpty && dateAnswer._2._2.isEmpty) {
          fail(
            s"because date matcher found ${dateAnswer._1.head.calendar.getTime} within ${dateAnswer._2._1} where None was expected")
        } else if (dateAnswer._1.isEmpty && dateAnswer._2._2.nonEmpty) {
          fail(s"because date matcher could not find anything within ${dateAnswer._2._1}")
        } else {
          val result = dateAnswer._1
          assert(
            result.head.calendar.get(Calendar.YEAR) == dateAnswer._2._2.get.get(Calendar.YEAR) &&
              result.head.calendar.get(Calendar.MONTH) == dateAnswer._2._2.get
                .get(Calendar.MONTH) &&
              result.head.calendar.get(Calendar.DAY_OF_MONTH) == dateAnswer._2._2.get
                .get(Calendar.DAY_OF_MONTH) &&
              result.head.calendar.get(Calendar.DAY_OF_WEEK) == dateAnswer._2._2.get
                .get(Calendar.DAY_OF_WEEK),
            s"because result ${result.head.calendar.getTime} is not expected ${dateAnswer._2._2.get.getTime} for string ${dateAnswer._2._1}")
        }
      }
    })

  "a MultiDateMatcher" should "ignore chunks of text with nothing relevant" taggedAs FastTest in {
    val _: Dataset[Row] = DataBuilder.multipleDataBuild(Array("2014/01/23", "day after tomorrow"))
  }

  "a MultiDateMatcher" should "be writable and readable" taggedAs FastTest in {
    val dateMatcher = new MultiDateMatcher().setOutputFormat("YYYY")
    val path = "./test-output-tmp/datematcher"
    dateMatcher.write.overwrite().save(path)
    val dateMatcherRead = MultiDateMatcher.read.load(path)
    assert(dateMatcherRead.getOutputFormat == dateMatcher.getOutputFormat)
  }

  "a MultiDateMatcher" should "correctly search for input formats to output format" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(
      Array(
        "Neighbouring Austria has already locked down its population this week for at until 2021/10/12, " +
          "becoming the first to reimpose such restrictions. It will also require the whole population to be " +
          "vaccinated from the second month of 2022, infuriating many in a country where scepticism about state mandates " +
          "affecting individual freedoms runs high in the next 02/2022."))

    val inputFormats = Array("yyyy/dd/MM", "yyyy", "MM/yyyy")
    val outputFormat = "yyyy/MM/dd"

    val date = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(1900)
      .setInputFormats(inputFormats)
      .setOutputFormat(outputFormat)
      .transform(data)

    val results = Annotation.collect(date, "date").flatten.toSeq.sortBy(_.end)

    val expectedDates = Seq(
      Annotation(DATE, 83, 86, "2021/01/01", Map("sentence" -> "0")),
      Annotation(DATE, 83, 92, "2021/12/10", Map("sentence" -> "0")),
      Annotation(DATE, 229, 232, "2022/01/01", Map("sentence" -> "0")),
      Annotation(DATE, 354, 361, "2022/02/01", Map("sentence" -> "0")))

    assert(results == expectedDates)
  }

  "a MultiDateMatcher" should "correctly disambiguating non dates with input formats provided" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(Array(
      "Omicron is a new variant of COVID-19, which the World Health Organization designated a " +
        "\"variant of concern\" on Nov. 26, 2021/26/11. The name comes from the letter in the Greek alphabet.\n\n" +
        "The omicron variant was first detected by scientists in South Africa, " +
        "where it is believed to be the cause of a recent spike in cases in the Gauteng province." +
        "More updates will be reported in 2022."))

    val inputFormats = Array("yyyy/dd/MM", "yyyy", "MM/yyyy")
    val outputFormat = "yyyy/MM/dd"

    val date = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(1900)
      .setInputFormats(inputFormats)
      .setOutputFormat(outputFormat)
      .transform(data)

    val results = Annotation.collect(date, "date").flatten.toSeq.sortBy(_.end)

    val expectedDates = Seq(
      Annotation(DATE, 120, 123, "2021/01/01", Map("sentence" -> "0")),
      Annotation(DATE, 120, 129, "2021/11/26", Map("sentence" -> "0")),
      Annotation(DATE, 378, 381, "2022/01/01", Map("sentence" -> "0")))

    assert(results == expectedDates)
  }

  "a MultiDateMatcher" should "correctly matches provided full input format" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(Array(
      "Omicron is a new variant of COVID-19, which the World Health Organization designated a " +
        "\"variant of concern\" on Nov. 26, 2021/26/11. The name comes from the letter in the Greek alphabet.\n\n" +
        "The omicron variant was first detected by scientists in South Africa, " +
        "where it is believed to be the cause of a recent spike in cases in the Gauteng province." +
        "More updates will be reported in 2022."))

    val inputFormats = Array("yyyy/dd/MM")
    val outputFormat = "yyyy/MM/dd"

    val date = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(1900)
      .setInputFormats(inputFormats)
      .setOutputFormat(outputFormat)
      .transform(data)

    val results = Annotation.collect(date, "date").flatten.toSeq.sortBy(_.end)

    val expectedDates = Seq(Annotation(DATE, 120, 129, "2021/11/26", Map("sentence" -> "0")))

    assert(results == expectedDates)
  }

  "a MultiDateMatcher" should "correctly matches provided year input format" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(Array(
      "Omicron is a new variant of COVID-19, which the World Health Organization designated a " +
        "\"variant of concern\" on Nov. 26, 2021/26/11. The name comes from the letter in the Greek alphabet.\n\n" +
        "The omicron variant was first detected by scientists in South Africa, " +
        "where it is believed to be the cause of a recent spike in cases in the Gauteng province." +
        "More updates will be reported in 2022."))

    val inputFormats = Array("yyyy")
    val outputFormat = "yyyy/MM/dd"

    val date = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(1900)
      .setInputFormats(inputFormats)
      .setOutputFormat(outputFormat)
      .transform(data)

    val results = Annotation.collect(date, "date").flatten.toSeq.sortBy(_.end)

    val expectedDates = Seq(
      Annotation(DATE, 120, 123, "2021/01/01", Map("sentence" -> "0")),
      Annotation(DATE, 378, 381, "2022/01/01", Map("sentence" -> "0")))

    assert(results == expectedDates)
  }

  "a MultiDateMatcher" should "correctly not match input formats" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(Array(
      "Omicron is a new variant of COVID-19, which the World Health Organization designated a " +
        "\"variant of concern\" on Nov. 26, 2021/26/11. The name comes from the letter in the Greek alphabet.\n\n" +
        "The omicron variant was first detected by scientists in South Africa, " +
        "where it is believed to be the cause of a recent spike in cases in the Gauteng province." +
        "More updates will be reported in 2022."))

    val inputFormats = Array("MM/yyyy")
    val outputFormat = "yyyy/MM/dd"

    val date = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(1900)
      .setInputFormats(inputFormats)
      .setOutputFormat(outputFormat)
      .transform(data)

    val results = Annotation.collect(date, "date").flatten.toSeq.sortBy(_.end)
    val expectedDates = Seq.empty

    assert(results == expectedDates)
  }

  "a MultiDateMatcher" should "correctly find all possible dates in a text" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(Array("""
          Lease Period	Monthly Installment of Base Rent
          January 1, 2021 –December 31, 2021	$20,304.85*
          January 1, 2022 –December 31, 2022	$20,914.00
      """))

    val dateMatcher = new MultiDateMatcher()
      .setInputCols(Array("document"))
      .setOutputCol("date")
      .setOutputFormat("yyyy/MM/dd")
      .setRelaxedFactoryStrategy(MatchStrategy.MATCH_ALL)
      .transform(data)

    val results = Annotation.collect(dateMatcher, "date").flatten.toSeq.sortBy(_.end)

    val expectedDates = Seq(
      Annotation(DATE, 67, 81, "2021/01/01", Map("sentence" -> "0")),
      Annotation(DATE, 84, 100, "2021/12/31", Map("sentence" -> "0")),
      Annotation(DATE, 103, 138, "2022/01/20", Map("sentence" -> "0")),
      Annotation(DATE, 132, 157, "2022/12/01", Map("sentence" -> "0")))

    assert(results == expectedDates)
  }

}
