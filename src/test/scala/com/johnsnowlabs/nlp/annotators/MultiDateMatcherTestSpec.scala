package com.johnsnowlabs.nlp.annotators

import java.util.Calendar

import com.johnsnowlabs.nlp.{AnnotatorType, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 5/10/2017.
  */
class MultiDateMatcherTestSpec extends FlatSpec with DateMatcherBehaviors {

  val dateMatcher = new MultiDateMatcher
  "a MultiDateMatcher" should s"be of type ${AnnotatorType.DATE}" in {
    assert(dateMatcher.outputAnnotatorType == AnnotatorType.DATE)
  }

  val dateData: Dataset[Row] = DataBuilder.multipleDataBuild(Array("2014/01/23", "day after tomorrow"))

  "A full MultiDateMatcher pipeline with some sentences" should behave like sparkBasedDateMatcher(dateData)

  val currentYear = Calendar.getInstance.get(Calendar.YEAR)
  val nextThursdayCalendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, 1)
    while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.THURSDAY) calendar.add(Calendar.DAY_OF_MONTH, 1)
    calendar
  }
  val lastWednesdayCalendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, -1)
    while (calendar.get(Calendar.DAY_OF_WEEK) != Calendar.WEDNESDAY) calendar.add(Calendar.DAY_OF_MONTH, -1)
    calendar
  }
  val tomorrowCalendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, 1)
    calendar
  }
  val yesterdayCalendar = {
    val calendar = Calendar.getInstance
    calendar.add(Calendar.DAY_OF_MONTH, -1)
    calendar
  }
  def nextCalendar(which: Int) = {
    val calendar = Calendar.getInstance
    calendar.add(which, 1)
    calendar
  }
  def setTimeTo(calendar: Calendar, hour: Int, minutes: Int, seconds: Int) = {
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
    ("1978-01-28", Some(new Calendar.Builder().setDate(1978, 1-1, 28).build)),
    ("1984/04/02", Some(new Calendar.Builder().setDate(1984, 4-1, 2).build)),
    ("1/02/1980", Some(new Calendar.Builder().setDate(1980, 1-1, 2).build)),
    ("2/28/79", Some(new Calendar.Builder().setDate(1979, 2-1, 28).build)),
    ("The 31st of April in the year 2008", Some(new Calendar.Builder().setDate(2008, 4-1, 31).build)),
    ("Fri, 21 Nov 1997", Some(new Calendar.Builder().setDate(1997, 11-1, 21).build)),
    ("Jan 21, '97", Some(new Calendar.Builder().setDate(1997, 1-1, 21).build)),
    ("Sun, Nov 21", Some(new Calendar.Builder().setDate(currentYear, 11-1, 21).build)),
    ("jan 1st", Some(new Calendar.Builder().setDate(currentYear, 1-1, 1).build)),
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
    ("day after", Some(tomorrowCalendar)),
    ("the day before", Some(yesterdayCalendar)),
    //"the monday after",
    //"the monday before"
    //NS: "2 fridays before",
    //NS: "4 tuesdays after"
    ("Let's meet on 20th of February.", Some(new Calendar.Builder().setDate(currentYear, 2-1, 20).build)),
    ("Today is March 14th 2019.", Some(new Calendar.Builder().setDate(2019, 3-1, 14).build)),
    ("10-02-19", Some(new Calendar.Builder().setDate(2019, 10-1, 2).build)),
    // Breaking use cases
    ("June 2015", Some(new Calendar.Builder().setDate(2015, 6-1, 1).build)),
    ("August 2016", Some(new Calendar.Builder().setDate(2016, 8-1, 1).build)),
    ("4", None),
    ("L-2", None),
    ("Tarceva", None),
    ("2", None),
    ("3", None),
    ("Xgeva", None),
    ("today 4", Some(Calendar.getInstance)),
    ("1 month", None)
  )

  dateSentences.map(date => dateMatcher.extractDate(date._1)).zip(dateSentences).foreach(dateAnswer => {
    "a MultiDateMatcher" should s"successfully parse ${dateAnswer._2._1} as ${dateAnswer._2._2.map(_.getTime)}" in {
      if (dateAnswer._1.isEmpty && dateAnswer._2._2.isEmpty)
        succeed
      else if (dateAnswer._1.nonEmpty && dateAnswer._2._2.isEmpty) {
        fail(s"because date matcher found ${dateAnswer._1.head.calendar.getTime} within ${dateAnswer._2._1} where None was expected")
      }
      else if (dateAnswer._1.isEmpty && dateAnswer._2._2.nonEmpty) {
        fail(s"because date matcher could not find anything within ${dateAnswer._2._1}")
      }
      else {
        val result = dateAnswer._1
        assert(
          result.head.calendar.get(Calendar.YEAR) == dateAnswer._2._2.get.get(Calendar.YEAR) &&
            result.head.calendar.get(Calendar.MONTH) == dateAnswer._2._2.get.get(Calendar.MONTH) &&
            result.head.calendar.get(Calendar.DAY_OF_MONTH) == dateAnswer._2._2.get.get(Calendar.DAY_OF_MONTH) &&
            result.head.calendar.get(Calendar.DAY_OF_WEEK) == dateAnswer._2._2.get.get(Calendar.DAY_OF_WEEK),
          s"because result ${result.head.calendar.getTime} is not expected ${dateAnswer._2._2.get.getTime} for string ${dateAnswer._2._1}")
      }
    }
  })

  "a MultiDateMatcher" should "ignore chunks of text with nothing relevant" in {
    val data: Dataset[Row] = DataBuilder.multipleDataBuild(Array("2014/01/23", "day after tomorrow"))
  }

  "a MultiDateMatcher" should "be writable and readable" in {
    val dateMatcher = new MultiDateMatcher().setFormat("YYYY")
    val path = "./test-output-tmp/datematcher"
    dateMatcher.write.overwrite().save(path)
    val dateMatcherRead = MultiDateMatcher.read.load(path)
    assert(dateMatcherRead.getFormat == dateMatcher.getFormat)
  }

}
