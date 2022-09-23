package com.johnsnowlabs.nlp.annotators.common

import org.json4s.jackson.JsonMethods.parse
import com.johnsnowlabs.util.JsonParser.formats
import org.json4s.jackson.Serialization.write
import org.sparkproject.dmg.pmml.True

import scala.util.Try

case class TableData(header: List[String], rows: List[List[String]]) {

  def toJson: String = {
    write(Map("header" -> header, "rows" -> rows))
  }

}

object TableData {

  val csvDelimiterEscapePattern = "(?=([^\"]*\"[^\"]*\")*[^\"]*$)"

  def fromJson(json: String): TableData = {
    Try(parse(json).extract[TableData]).getOrElse(Empty)
  }

  def fromCsv(
      csv: String,
      delimiter: String = ",",
      escapeDelimiterByDoubleQuotes: Boolean = true): TableData = {
    val delimiterPattern = delimiter + (if (escapeDelimiterByDoubleQuotes) csvDelimiterEscapePattern else "")

    // trimming function which preserves tabs
    def trimSpace(string: String): String = {
      string
        .stripPrefix(" ")
        .stripSuffix(" ")
        .stripPrefix("\n")
        .stripSuffix("\n")
    }

    def splitRow(row: String): List[String] = {
      (row + " ")
        .split(delimiterPattern)
        .map(trimSpace)
        .map(x =>
          if (escapeDelimiterByDoubleQuotes)
            if (x.startsWith("\"") && x.endsWith("\""))
              trimSpace(x.slice(1, x.length - 1).replace("\"\"", "\""))
            else x
          else x)
        .toList
    }

    try {
      val lines = trimSpace(csv).split("\n").map(trimSpace)
      val header = splitRow(lines.head)
      val rows = lines.tail
        .map(x => {
          val row = splitRow(x)
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
