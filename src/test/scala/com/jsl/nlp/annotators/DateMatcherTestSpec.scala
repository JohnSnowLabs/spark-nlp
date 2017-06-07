package com.jsl.nlp.annotators

import java.util.Calendar

import com.jsl.nlp.DataBuilder
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 5/10/2017.
  */
class DateMatcherTestSpec extends FlatSpec with DateMatcherBehaviors {

  val dateMatcher = new DateMatcher
  "a DateMatcher" should s"be of type ${DateMatcher.aType}" in {
    assert(dateMatcher.aType == DateMatcher.aType)
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
    ("the day before", yesterdayCalendar)
    //"the monday after",
    //"the monday before"
    //NS: "2 fridays before",
    //NS: "4 tuesdays after"
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

}
