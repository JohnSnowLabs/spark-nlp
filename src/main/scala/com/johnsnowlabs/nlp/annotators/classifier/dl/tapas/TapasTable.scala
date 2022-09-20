package com.johnsnowlabs.nlp.annotators.classifier.dl.tapas

import scala.util.Try

case class TapasCellDate(day: Option[Int], month: Option[Int], year: Option[Int]) {
  def isDefined: Boolean = {
    day.isDefined || month.isDefined || year.isDefined
  }

  def compareTo(other: TapasCellDate): Boolean = TapasCellDate.compare(this, other)

  def isEqualTo(other: TapasCellDate): Boolean = TapasCellDate.areEqual(this, other)
}

object TapasCellDate {
  def Empty(): TapasCellDate  = {
    TapasCellDate(None, None, None)
  }

  def computeNumericRelation(v1: TapasCellDate, v2: TapasCellDate): Int = {
    if (v1.isDefined && v2.isDefined){
      if (areEqual(v1, v2))
        TapasNumericRelation.EQ
      else
        if (compare(v1, v2)) TapasNumericRelation.LT else TapasNumericRelation.GT

    } else TapasNumericRelation.NONE
  }

  def areEqual(v1: TapasCellDate, v2: TapasCellDate): Boolean = {
    val dayMatch = v1.day.getOrElse(Int.MinValue) == v2.day.getOrElse(Int.MinValue)
    val monthMatch = v1.month.getOrElse(Int.MinValue) == v2.month.getOrElse(Int.MinValue)
    val yearMatch = v1.year.getOrElse(Int.MinValue) == v2.year.getOrElse(Int.MinValue)
    dayMatch && monthMatch && yearMatch
  }

  def compare(v1: TapasCellDate, v2: TapasCellDate): Boolean = {
    val dayDiff = Try(v1.day.get - v2.day.get).getOrElse(0)
    val monthDiff = Try(v1.month.get  - v2.month.get).getOrElse(0)
    val yearDiff = Try(v1.year.get - v2.year.get).getOrElse(0)

    (dayDiff < 0) || (monthDiff < 0) || (yearDiff < 0)
  }
}

case class TapasCellValue(number: Option[Float], date: TapasCellDate) {
  def compareTo(other: TapasCellValue): Boolean = TapasCellValue.compare(this, other)
  def getNumericRelationTo(other: TapasCellValue): Int = TapasCellValue.computeNumericRelation(this, other)
}

object TapasCellValue {

  def compareNumbers(v1: TapasCellValue, v2: TapasCellValue): Boolean = {
    val n1 = v1.number.getOrElse(Float.MinValue)
    val n2 = v2.number.getOrElse(Float.MinValue)
    n1 < n2
  }

  def compareDates(v1: TapasCellValue, v2: TapasCellValue): Boolean = v1.date.compareTo(v2.date)

  def compare(v1: TapasCellValue, v2: TapasCellValue): Boolean = {
    if (v1.number.isDefined && v2.number.isDefined){
      compareNumbers(v1, v2)
    } else if(v1.date.isDefined && v1.date.isDefined) {
      compareDates(v1, v2)
    } else {
      false
    }
  }

  def computeNumericRelation(v1: TapasCellValue, v2: TapasCellValue): Int = {
    if (v1.number.isDefined && v2.number.isDefined) (v1.number.get  - v2.number.get) match {
      case 0 => return TapasNumericRelation.EQ
      case x if x > 0 => return TapasNumericRelation.GT
      case x if x < 0 => return TapasNumericRelation.LT
    } else if (v1.date.isDefined && v2.date.isDefined) {
      return TapasCellDate.computeNumericRelation(v1.date, v2.date)
    }

    TapasNumericRelation.NONE
  }
}

case class TapasNumericValueSpan(begin: Int, end: Int, value: TapasCellValue) {

  def valueId: String = {
    if (value.number.isDefined)
      value.number.toString
    else if (value.date.day.isDefined || value.date.month.isDefined || value.date.year.isDefined)
      Array(
        if (value.date.day.isDefined) value.date.day.get.toString else "NA",
        if (value.date.month.isDefined) value.date.month.get.toString else "NA",
        if (value.date.year.isDefined) value.date.year.get.toString else "NA"
      ).mkString("@")
    else TapasNumericValueSpan.emptyValueId
  }

  def compareTo(other: TapasNumericValueSpan): Boolean = TapasCellValue.compare(this.value, other.value)
}

object TapasNumericValueSpan {
  val emptyValueId = "NA"
}

case class TapasTable(header: Array[String], rows: Array[Array[String]])