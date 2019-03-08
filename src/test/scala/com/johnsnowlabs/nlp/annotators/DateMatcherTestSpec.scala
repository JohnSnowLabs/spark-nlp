package com.johnsnowlabs.nlp.annotators

import java.util.Calendar

import com.johnsnowlabs.nlp.{AnnotatorType, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 5/10/2017.
  */
class DateMatcherTestSpec extends FlatSpec with DateMatcherBehaviors {

  val dateMatcher = new DateMatcher
  "a DateMatcher" should s"be of type ${AnnotatorType.DATE}" in {
    assert(dateMatcher.outputAnnotatorType == AnnotatorType.DATE)
  }

  val dateData: Dataset[Row] = DataBuilder.multipleDataBuild(Array("2014/01/23", "day after tomorrow"))

  "A full DateMatcher pipeline with some sentences" should behave like sparkBasedDateMatcher(dateData)

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

  val dateSentences = Array(
    ("1978-01-28", new Calendar.Builder().setDate(1978, 1-1, 28).build),
    ("1984/04/02", new Calendar.Builder().setDate(1984, 4-1, 2).build),
    ("1/02/1980", new Calendar.Builder().setDate(1980, 1-1, 2).build),
    ("2/28/79", new Calendar.Builder().setDate(1979, 2-1, 28).build),
    ("The 31st of April in the year 2008", new Calendar.Builder().setDate(2008, 4-1, 31).build),
    ("Fri, 21 Nov 1997", new Calendar.Builder().setDate(1997, 11-1, 21).build),
    ("Jan 21, '97", new Calendar.Builder().setDate(1997, 1-1, 21).build),
    ("Sun, Nov 21", new Calendar.Builder().setDate(currentYear, 11-1, 21).build),
    ("jan 1st", new Calendar.Builder().setDate(currentYear, 1-1, 1).build),
    //NS: "february twenty-eighth",
    ("next thursday", nextThursdayCalendar),
    ("last wednesday", lastWednesdayCalendar),
    ("today", Calendar.getInstance),
    ("tomorrow", tomorrowCalendar),
    ("yesterday", yesterdayCalendar),
    ("next week", nextCalendar(Calendar.WEEK_OF_MONTH)),
    ("next month", nextCalendar(Calendar.MONTH)),
    ("next year", nextCalendar(Calendar.YEAR)),
    //NS: "3 days from now",
    //NS: "three weeks ago",
    ("day after", tomorrowCalendar),
    ("the day before", yesterdayCalendar),
    //"the monday after",
    //"the monday before"
    //NS: "2 fridays before",
    //NS: "4 tuesdays after"
    ("0600h", setTimeTo(Calendar.getInstance, 6,0,0)),
    ("06:00 hours", setTimeTo(Calendar.getInstance, 6,0,0)),
    ("6pm", setTimeTo(Calendar.getInstance, 18,0,0)),
    ("5:30 a.m.", setTimeTo(Calendar.getInstance, 5,30,0)),
    ("at 5", setTimeTo(Calendar.getInstance, 17,0,0)),
    ("12:59", setTimeTo(Calendar.getInstance, 12,59,0)),
    ("23:59", setTimeTo(Calendar.getInstance, 23,59,0)),
    ("1988/11/23 6pm", setTimeTo(new Calendar.Builder().setDate(1988, 11-1, 23).build, 18, 0, 0)),
    ("next week at 7.30", setTimeTo(nextCalendar(Calendar.WEEK_OF_MONTH), 19, 0, 0)),
    ("5 am tomorrow", setTimeTo(tomorrowCalendar, 5, 0, 0))
  )

  dateSentences.map(date => dateMatcher.extractDate(date._1)).zip(dateSentences).foreach(dateAnswer => {
    "a DateMatcher" should s"successfully parse ${dateAnswer._2._1}" in {
      val result = dateAnswer._1.getOrElse(fail(s"because we could not parse ${dateAnswer._2._1}"))
      assert(
        result.calendar.get(Calendar.YEAR) == dateAnswer._2._2.get(Calendar.YEAR) &&
          result.calendar.get(Calendar.MONTH) == dateAnswer._2._2.get(Calendar.MONTH) &&
          result.calendar.get(Calendar.DAY_OF_MONTH) == dateAnswer._2._2.get(Calendar.DAY_OF_MONTH) &&
          result.calendar.get(Calendar.DAY_OF_WEEK) == dateAnswer._2._2.get(Calendar.DAY_OF_WEEK),
        s"because result ${result.calendar.getTime} is not expected ${dateAnswer._2._2.getTime}")
    }
  })

  "a DateMatcher" should "be writable and readable" in {
    val dateMatcher = new DateMatcher().setFormat("YYYY")
    val path = "./test-output-tmp/datematcher"
    dateMatcher.write.overwrite().save(path)
    val dateMatcherRead = DateMatcher.read.load(path)
    assert(dateMatcherRead.getFormat == dateMatcher.getFormat)
  }

}
