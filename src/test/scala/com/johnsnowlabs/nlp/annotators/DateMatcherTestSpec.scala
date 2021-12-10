/*
 * Copyright 2017-2021 John Snow Labs
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
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DataBuilder}
import com.johnsnowlabs.tags.FastTest

import org.apache.spark.sql.{Dataset, Row}

import org.scalatest.flatspec.AnyFlatSpec

import java.util.Calendar


class DateMatcherTestSpec extends AnyFlatSpec with DateMatcherBehaviors {

  val dateMatcher = new DateMatcher
  "a DateMatcher" should s"be of type ${AnnotatorType.DATE}" taggedAs FastTest in {
    assert(dateMatcher.outputAnnotatorType == AnnotatorType.DATE)
  }

  val dateData: Dataset[Row] = DataBuilder.multipleDataBuild(Array("2014/01/23", "day after tomorrow"))

  "A full DateMatcher pipeline with some sentences" should behave like sparkBasedDateMatcher(dateData)

  val currentYear: Int = Calendar.getInstance.get(Calendar.YEAR)
  val nextThursdayCalendar: Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, 1)
    while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.THURSDAY) calendar.add(Calendar.DAY_OF_MONTH, 1)
    calendar
  }
  val lastWednesdayCalendar: Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, -1)
    while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.WEDNESDAY) calendar.add(Calendar.DAY_OF_MONTH, -1)
    calendar
  }
  val tomorrowCalendar: Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, 1)
    calendar
  }

  val afterTomorrowCalendar: Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, 2)
    calendar
  }
  val yesterdayCalendar: Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, -1)
    calendar
  }

  val beforeYesterdayCalendar: Calendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, -2)
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
      calendar.get(Calendar.DAY_OF_MONTH)
    )
    calendarBuild.setTimeOfDay(hour, minutes, seconds)
    calendarBuild.build
  }

  val dateSentences: Array[(String, Option[Calendar])] = Array(
    ("1978-01-28", Some(new Calendar.Builder().setDate(1978, 1 - 1, 28).build)),
    ("1984/04/02", Some(new Calendar.Builder().setDate(1984, 4 - 1, 2).build)),
    ("1/02/1980", Some(new Calendar.Builder().setDate(1980, 1 - 1, 2).build)),
    ("2/28/79", Some(new Calendar.Builder().setDate(1979, 2 - 1, 28).build)),
    ("The 31st of April in the year 2008", Some(new Calendar.Builder().setDate(2008, 4 - 1, 31).build)),
    ("Fri, 21 Nov 1997", Some(new Calendar.Builder().setDate(1997, 11 - 1, 21).build)),
    ("Jan 21, '97", Some(new Calendar.Builder().setDate(1997, 1 - 1, 21).build)),
    ("Sun, Nov 21", Some(new Calendar.Builder().setDate(currentYear, 11 - 1, 21).build)),
    ("jan 1st", Some(new Calendar.Builder().setDate(currentYear, 1 - 1, 1).build)),
    //NS: "february twenty-eighth",
    ("next thursday", Some(nextThursdayCalendar)),
    ("last wednesday", Some(lastWednesdayCalendar)),
    ("today", Some(Calendar.getInstance)),
    ("tomorrow", Some(tomorrowCalendar)),
    ("yesterday", Some(yesterdayCalendar)),
    ("next week", Some(nextCalendar(Calendar.WEEK_OF_MONTH))),
    ("next month", Some(nextCalendar(Calendar.MONTH))),
    ("next year", Some(nextCalendar(Calendar.YEAR))),
    //NS: "3 days from now",
    //NS: "three weeks ago",
    (" day after", Some(tomorrowCalendar)),
    (" day after tomorrow", Some(afterTomorrowCalendar)),
    ("the day before", Some(yesterdayCalendar)),
    ("the day before yesterday", Some(beforeYesterdayCalendar)),
    //"the monday after",
    //"the monday before"
    //NS: "2 fridays before",
    //NS: "4 tuesdays after"
    ("0600h", Some(setTimeTo(Calendar.getInstance, 6, 0, 0))),
    ("06:00 hours", Some(setTimeTo(Calendar.getInstance, 6, 0, 0))),
    ("6pm", Some(setTimeTo(Calendar.getInstance, 18, 0, 0))),
    ("5:30 a.m.", Some(setTimeTo(Calendar.getInstance, 5, 30, 0))),
    ("at 5", Some(setTimeTo(Calendar.getInstance, 17, 0, 0))),
    ("12:59", Some(setTimeTo(Calendar.getInstance, 12, 59, 0))),
    ("23:59", Some(setTimeTo(Calendar.getInstance, 23, 59, 0))),
    ("1988/11/23 6pm", Some(setTimeTo(new Calendar.Builder().setDate(1988, 11 - 1, 23).build, 18, 0, 0))),
    ("next week at 7.30", Some(setTimeTo(nextCalendar(Calendar.WEEK_OF_MONTH), 19, 0, 0))),
    ("5 am tomorrow", Some(setTimeTo(tomorrowCalendar, 5, 0, 0))),
    ("Let's meet on 20th of February.", Some(new Calendar.Builder().setDate(currentYear, 2 - 1, 20).build)),
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
    ("1 month", None),
    ("07/2015", Some(new Calendar.Builder().setDate(2015, 7 - 1, 1).build))
  )

  dateSentences.map(date => dateMatcher.extractDate(date._1)).zip(dateSentences).foreach(dateAnswer => {
    "a DateMatcher" should s"successfully parse ${dateAnswer._2._1} as ${dateAnswer._2._2.map(_.getTime)}" taggedAs FastTest in {
      if (dateAnswer._1.isEmpty && dateAnswer._2._2.isEmpty)
        succeed
      else if (dateAnswer._1.nonEmpty && dateAnswer._2._2.isEmpty) {
        fail(s"because date matcher found ${dateAnswer._1.get.calendar.getTime} within ${dateAnswer._2._1} where None was expected")
      }
      else if (dateAnswer._1.isEmpty && dateAnswer._2._2.nonEmpty) {
        fail(s"because date matcher could not find anything within ${dateAnswer._2._1}")
      }
      else {
        val result = dateAnswer._1.getOrElse(fail(s"because we could not parse ${dateAnswer._2._1}"))
        assert(
          result.calendar.get(Calendar.YEAR) == dateAnswer._2._2.get.get(Calendar.YEAR) &&
            result.calendar.get(Calendar.MONTH) == dateAnswer._2._2.get.get(Calendar.MONTH) &&
            result.calendar.get(Calendar.DAY_OF_MONTH) == dateAnswer._2._2.get.get(Calendar.DAY_OF_MONTH) &&
            result.calendar.get(Calendar.DAY_OF_WEEK) == dateAnswer._2._2.get.get(Calendar.DAY_OF_WEEK),
          s"because result ${result.calendar.getTime} is not expected ${dateAnswer._2._2.get.getTime} for string ${dateAnswer._2._1}")
      }
    }
  })

  "a DateMatcher" should "ignore chunks of text with nothing relevant" taggedAs FastTest in {
    val data: Dataset[Row] = DataBuilder.multipleDataBuild(Array("2014/01/23", "day after tomorrow"))
  }

  "a DateMatcher" should "correctly use anchorDate params for relative dates" taggedAs FastTest in {
    val data: Dataset[Row] = DataBuilder.multipleDataBuild(Array("2014/01/23", "see you a day after"))

    val expectedDates = Seq(
      Annotation(DATE, 0, 9, "2014/01/23", Map("sentence" -> "0")),
      Annotation(DATE, 9, 18, "2020/01/12", Map("sentence" -> "0"))
    )

    val date = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(2020)
      .setAnchorDateMonth(1)
      .setAnchorDateDay(11)
      .transform(data)

    val results = Annotation.collect(date, "date").flatten.toSeq
    assert(results == expectedDates)
  }

  "a DateMatcher" should "be writable and readable" taggedAs FastTest in {
    val dateMatcher = new DateMatcher().setOutputFormat("YYYY")
    val path = "./test-output-tmp/datematcher"
    dateMatcher.write.overwrite().save(path)
    val dateMatcherRead = DateMatcher.read.load(path)
    assert(dateMatcherRead.getOutputFormat == dateMatcher.getOutputFormat)
  }

  "a DateMatcher" should "correctly disambiguate the monthly sub-words in text" taggedAs FastTest in {
    val data: Dataset[Row] = DataBuilder.multipleDataBuild(
      Array("right over-the-needle catheter system 18 gauge;1 1/2 in length")
    )

    val date = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(1900)
      .setOutputFormat("yyyy/MM/dd")
      .transform(data)

    val results = Annotation.collect(date, "date").flatten.toSeq

    val defaultYearWhenMissing = Calendar.getInstance.get(Calendar.YEAR)
    val defaultMonthWhenMissing = "01"

    val expectedDates = Seq(
      Annotation(DATE, 38, 44, s"$defaultYearWhenMissing/$defaultMonthWhenMissing/18", Map("sentence" -> "0")))

    assert(results == expectedDates)
  }

  "a DateMatcher" should "correctly search for input formats to output format" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(
      Array("Neighbouring Austria has already locked down its population this week for at until 2021/10/12, " +
        "becoming the first to reimpose such restrictions. It will also require the whole population to be " +
        "vaccinated from the second month of 2022, infuriating many in a country where scepticism about state mandates " +
        "affecting individual freedoms runs high in the next 01-22.")
    )

    val inputFormats = Array("yyyy/dd/MM")
    val outputFormat = "yyyy/MM/dd"

    val date = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(1900)
      .setInputFormats(inputFormats)
      .setOutputFormat(outputFormat)
      .transform(data)

    val results = Annotation.collect(date, "date").flatten.toSeq

    val expectedDates = Seq(
      Annotation(DATE, 83, 92, "2021/12/10", Map("sentence" -> "0")))

    assert(results == expectedDates)
  }

  "a DateMatcher" should "correctly disambiguating non dates with input formats provided" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(
      Array("Omicron is a new variant of COVID-19, which the World Health Organization designated a " +
        "\"variant of concern\" on Nov. 26, 2021/26/11. The name comes from the letter in the Greek alphabet.\n\n" +
        "The omicron variant was first detected by scientists in South Africa, " +
        "where it is believed to be the cause of a recent spike in cases in the Gauteng province." +
        "More updates will be reported in 2022."))

    val inputFormats = Array("yyyy/dd/MM", "yyyy", "MM/yyyy")
    val outputFormat = "yyyy/MM/dd"

    val date = new DateMatcher()
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

  "a DateMatcher" should "correctly match sorted input formats" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(
      Array("Omicron is a new variant of COVID-19, which the World Health Organization designated a " +
        "\"variant of concern\" on Nov. 26, 2021/26/11. The name comes from the letter in the Greek alphabet.\n\n" +
        "The omicron variant was first detected by scientists in South Africa, " +
        "where it is believed to be the cause of a recent spike in cases in the Gauteng province." +
        "More updates will be reported in 2022."))

    val inputFormats = Array("yyyy", "yyyy/dd/MM", "MM/yyyy")
    val outputFormat = "yyyy/MM/dd"

    val date = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(1900)
      .setInputFormats(inputFormats)
      .setOutputFormat(outputFormat)
      .transform(data)

    val results = Annotation.collect(date, "date").flatten.toSeq.sortBy(_.end)

    val expectedDates = Seq(Annotation(DATE, 120, 123, "2021/01/01", Map("sentence" -> "0")))

    assert(results == expectedDates)
  }

  "a DateMatcher" should "correctly not match input formats" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.multipleDataBuild(
      Array("Omicron is a new variant of COVID-19, which the World Health Organization designated a " +
        "\"variant of concern\" on Nov. 26, 2021/26/11. The name comes from the letter in the Greek alphabet.\n\n" +
        "The omicron variant was first detected by scientists in South Africa, " +
        "where it is believed to be the cause of a recent spike in cases in the Gauteng province."))

    val inputFormats = Array("MM/yyyy")
    val outputFormat = "yyyy/MM/dd"

    val date = new DateMatcher()
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
}
