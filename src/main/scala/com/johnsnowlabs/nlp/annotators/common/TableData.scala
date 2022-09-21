package com.johnsnowlabs.nlp.annotators.common

import org.json4s.JsonAST
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import com.johnsnowlabs.util.JsonParser.formats

case class TableData(header: List[String], rows: List[List[String]]) {

  def toJSON: String = {
    compact(render(("header" -> header) ~ ("rows" -> rows)))
  }

}

object TableData {

  def fromJSON(json: String): TableData = {
    parse(json).extract[TableData]
  }

  def fromCSV(csv: String, delimiter: String): TableData = {
    val lines = csv.split("\n").map(_.trim)
    TableData(
      header = lines.head.split(delimiter).toList,
      rows = lines.tail.map(_.split(delimiter).toList).toList,
    )
  }

}
