package com.johnsnowlabs.nlp.annotators.common

import org.json4s.jackson.JsonMethods.parse
import com.johnsnowlabs.util.JsonParser.formats
import org.json4s.jackson.Serialization.write

import scala.util.Try

case class TableData(header: List[String], rows: List[List[String]]) {

  def toJson: String = {
    write(Map("header" -> header, "rows" -> rows))
  }

}

object TableData {

  def fromJson(json: String): TableData = {
    Try(parse(json).extract[TableData]).getOrElse(Empty)
  }

  def fromCsv(csv: String, delimiter: String): TableData = {
    try {
      val lines = csv.split("\n").map(_.trim)

      val header = lines.head.split(delimiter).toList
      val rows = lines.tail
        .map(x => {
          val row = x.split(delimiter).toList
          if (header.length == row.length)
            row
          else
            List()
        })
        .filter(_.nonEmpty)
        .toList

      TableData(header = header, rows = rows)
    } catch {
      case _: Exception => TableData.Empty
    }

  }

  def Empty: TableData = TableData(List(), List())

}
