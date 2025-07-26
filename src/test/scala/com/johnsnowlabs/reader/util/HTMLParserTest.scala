package com.johnsnowlabs.reader.util

import org.json4s.jackson.{JsonMethods, Serialization}
import org.json4s.{JValue, NoTypeHints}
import org.scalatest.flatspec.AnyFlatSpec

class HTMLParserTest extends AnyFlatSpec {

  private val PtToPx = 1.333

  "extractFontSize" should "extract px values correctly" in {
    assert(HTMLParser.extractFontSize("font-size: 18px;") == 18)
    assert(HTMLParser.extractFontSize("font-size : 24 px ;") == 24)
  }

  it should "extract pt values and convert to px" in {
    assert(HTMLParser.extractFontSize("font-size: 12pt;") == Math.round(12 * PtToPx).toInt)
    assert(HTMLParser.extractFontSize("font-size: 14 pt;") == Math.round(14 * PtToPx).toInt)
  }

  it should "extract em values using default baseEmPx" in {
    assert(HTMLParser.extractFontSize("font-size: 2em;") == 32) // 2 * 16
    assert(HTMLParser.extractFontSize("font-size: 1.08em;") == Math.round(1.08 * 16).toInt)
  }

  it should "extract rem values using default baseRemPx" in {
    assert(HTMLParser.extractFontSize("font-size: 1.5rem;") == 24) // 1.5 * 16
    assert(HTMLParser.extractFontSize("font-size: 0.9rem;") == Math.round(0.9 * 16).toInt)
  }

  it should "extract percent values using default parentPx" in {
    assert(HTMLParser.extractFontSize("font-size: 200%;") == 32) // 200% of 16
    assert(HTMLParser.extractFontSize("font-size: 75%;") == 12)
  }

  it should "allow overriding baseEmPx, baseRemPx, and parentPx" in {
    assert(HTMLParser.extractFontSize("font-size: 2em;", baseEmPx = 20) == 40)
    assert(HTMLParser.extractFontSize("font-size: 1.5rem;", baseRemPx = 10) == 15)
    assert(HTMLParser.extractFontSize("font-size: 50%;", parentPx = 10) == 5)
  }

  it should "return 0 for missing or unrecognized font-size" in {
    assert(HTMLParser.extractFontSize("font-weight: bold;") == 0)
    assert(HTMLParser.extractFontSize("font-size: large;") == 0)
    assert(HTMLParser.extractFontSize("") == 0)
  }

  it should "handle spaces and mixed case units" in {
    assert(HTMLParser.extractFontSize("font-size :  18PX ;") == 18)
    assert(HTMLParser.extractFontSize("font-size:  12Pt ;") == Math.round(12 * PtToPx).toInt)
    assert(HTMLParser.extractFontSize("font-size : 1.2eM ;") == Math.round(1.2 * 16).toInt)
  }

  // --- BOLD detection ---
  "isFormattedAsTitle" should "detect 'font-weight:bold' as title" in {
    assert(HTMLParser.isFormattedAsTitle("font-weight:bold;", 16))
    assert(HTMLParser.isFormattedAsTitle("font-weight: bold ;", 16))
    assert(HTMLParser.isFormattedAsTitle(" FONT-WEIGHT:BOLD ; ", 16)) // Mixed case
    assert(
      HTMLParser.isFormattedAsTitle("font-weight:bold;font-size:10px;", 16)
    ) // Bold but small font
  }

  it should "detect 'font-weight:bolder' as title" in {
    assert(HTMLParser.isFormattedAsTitle("font-weight:bolder;", 16))
    assert(HTMLParser.isFormattedAsTitle("font-weight: bolder ; font-size:12px;", 16))
  }

  it should "detect numeric bold values" in {
    assert(HTMLParser.isFormattedAsTitle("font-weight:700;", 16))
    assert(HTMLParser.isFormattedAsTitle("font-weight: 900 ;", 16))
    assert(HTMLParser.isFormattedAsTitle("font-weight:800; font-size:10px;", 16))
  }

  it should "not detect normal or light weights as title" in {
    assert(!HTMLParser.isFormattedAsTitle("font-weight:400;", 16))
    assert(!HTMLParser.isFormattedAsTitle("font-weight:normal;", 16))
    assert(!HTMLParser.isFormattedAsTitle("font-weight:light;", 16))
    assert(!HTMLParser.isFormattedAsTitle("font-weight:100;", 16))
  }

  // --- LARGE FONT detection ---
  it should "detect large font-size as title" in {
    assert(HTMLParser.isFormattedAsTitle("font-size: 20px;", 16))
    assert(HTMLParser.isFormattedAsTitle("font-size: 1.5em;", 16)) // 24px
    assert(!HTMLParser.isFormattedAsTitle("font-size: 12px;", 16))
  }

  // --- CENTERED TEXT detection ---
  it should "detect centered bold text as title" in {
    assert(HTMLParser.isFormattedAsTitle("font-weight:bold; text-align:center;", 16))
    assert(HTMLParser.isFormattedAsTitle("text-align:center; font-weight:bold;", 16))
    assert(HTMLParser.isFormattedAsTitle("font-weight:700; text-align:center;", 16))
  }

  it should "detect centered large text as title" in {
    assert(HTMLParser.isFormattedAsTitle("text-align:center; font-size:20px;", 16))
    assert(!HTMLParser.isFormattedAsTitle("text-align:center; font-size:12px;", 16))
  }

  // --- NEGATIVE CASES ---
  it should "return false for unrelated or empty styles" in {
    assert(!HTMLParser.isFormattedAsTitle("font-size:12px;", 16))
    assert(!HTMLParser.isFormattedAsTitle("text-align:left;", 16))
    assert(!HTMLParser.isFormattedAsTitle("", 16))
    assert(!HTMLParser.isFormattedAsTitle("font-style:italic;", 16))
  }

  // --- MIXED & EDGE CASES ---
  it should "handle mixed cases and excessive whitespace" in {
    assert(HTMLParser.isFormattedAsTitle("  FONT-WEIGHT:  BOLD  ;  ", 16))
    assert(HTMLParser.isFormattedAsTitle("font-size : 18PX ;", 16))
    assert(!HTMLParser.isFormattedAsTitle("font-size : 10PX ;", 16))
  }

  it should "detect bold and centered even if font-size is too small" in {
    assert(
      HTMLParser.isFormattedAsTitle("font-weight:bold; text-align:center; font-size:10px;", 16))
  }

  it should "detect large and centered even if not bold" in {
    assert(HTMLParser.isFormattedAsTitle("font-size:22px; text-align:center;", 16))
  }

  implicit val formats = Serialization.formats(NoTypeHints)

  "tableElementToJson" should "handle a table with caption, header, and multiple rows" in {
    val htmlTable =
      """
      <table border="1">
        <caption>Student Grades</caption>
        <tr>
          <th>Name</th>
          <th>Subject</th>
          <th>Grade</th>
        </tr>
        <tr>
          <td>Alice</td>
          <td>Math</td>
          <td>A</td>
        </tr>
        <tr>
          <td>Bob</td>
          <td>Science</td>
          <td>B+</td>
        </tr>
      </table>
      """
    val tableElement = HTMLParser.parseFirstTableElement(htmlTable)
    val jsonString = HTMLParser.tableElementToJson(tableElement)
    val parsedJson: JValue = parseJsonToJValue(jsonString)

    val caption: String = (parsedJson \ "caption").extract[String]
    val header: List[String] = (parsedJson \ "header").extract[List[String]]
    val rows: List[List[String]] = (parsedJson \ "rows").extract[List[List[String]]]

    assert(caption == "Student Grades")
    assert(header == List("Name", "Subject", "Grade"))
    assert(rows == List(List("Alice", "Math", "A"), List("Bob", "Science", "B+")))
  }

  it should "handle a table without caption" in {
    val htmlTable =
      """
      <table>
        <tr><th>City</th><th>Population</th></tr>
        <tr><td>NYC</td><td>8M</td></tr>
      </table>
      """
    val tableElement = HTMLParser.parseFirstTableElement(htmlTable)
    val jsonString = HTMLParser.tableElementToJson(tableElement)
    val parsedJson: JValue = parseJsonToJValue(jsonString)

    val caption: String = (parsedJson \ "caption").extract[String]
    val header: List[String] = (parsedJson \ "header").extract[List[String]]
    val rows: List[List[String]] = (parsedJson \ "rows").extract[List[List[String]]]

    assert(caption == "")
    assert(header == List("City", "Population"))
    assert(rows == List(List("NYC", "8M")))
  }

  it should "handle a table with no header row" in {
    val htmlTable =
      """
      <table>
        <tr><td>1</td><td>2</td></tr>
        <tr><td>3</td><td>4</td></tr>
      </table>
      """
    val tableElement = HTMLParser.parseFirstTableElement(htmlTable)
    val jsonString = HTMLParser.tableElementToJson(tableElement)
    val parsedJson: JValue = parseJsonToJValue(jsonString)

//    val header: List[String] = (parsedJson \ "header").extract[List[String]]
    val rows: List[List[String]] = (parsedJson \ "rows").extract[List[List[String]]]

    assert(rows.nonEmpty)
  }

  it should "handle a table with missing or empty cells" in {
    val htmlTable =
      """
      <table>
        <tr><th>A</th><th>B</th><th>C</th></tr>
        <tr><td>1</td><td>2</td></tr>
        <tr><td>3</td><td>4</td><td>5</td></tr>
      </table>
      """
    val tableElement = HTMLParser.parseFirstTableElement(htmlTable)
    val jsonString = HTMLParser.tableElementToJson(tableElement)
    val parsedJson: JValue = parseJsonToJValue(jsonString)

    val rows: List[List[String]] = (parsedJson \ "rows").extract[List[List[String]]]
    assert(rows.head.length == 2 || rows.head.length == 3)
    assert(rows(1).length == 3)
  }

  it should "handle an empty table" in {
    val htmlTable = "<table></table>"
    val tableElement = HTMLParser.parseFirstTableElement(htmlTable)
    val jsonString = HTMLParser.tableElementToJson(tableElement)
    val parsedJson: JValue = parseJsonToJValue(jsonString)

    val header: List[String] = (parsedJson \ "header").extract[List[String]]
    val rows: List[List[String]] = (parsedJson \ "rows").extract[List[List[String]]]

    assert(header.isEmpty)
    assert(rows.isEmpty)
  }

  it should "handle a table with nested tags in cells" in {
    val htmlTable =
      """
      <table>
        <tr><th>Name</th><th>Info</th></tr>
        <tr><td><b>Alice</b></td><td><span>Student</span></td></tr>
      </table>
      """
    val tableElement = HTMLParser.parseFirstTableElement(htmlTable)
    val jsonString = HTMLParser.tableElementToJson(tableElement)
    val parsedJson: JValue = parseJsonToJValue(jsonString)

    val rows: List[List[String]] = (parsedJson \ "rows").extract[List[List[String]]]
    assert(rows == List(List("Alice", "Student")))
  }

  it should "handle a table with excessive whitespace and newlines" in {
    val htmlTable =
      """
      <table>
        <tr>
          <th> X </th>
          <th> Y </th>
        </tr>
        <tr>
          <td>
            10
          </td>
          <td>
            20
          </td>
        </tr>
      </table>
      """
    val tableElement = HTMLParser.parseFirstTableElement(htmlTable)
    val jsonString = HTMLParser.tableElementToJson(tableElement)
    val parsedJson: JValue = parseJsonToJValue(jsonString)

    val header: List[String] = (parsedJson \ "header").extract[List[String]]
    val rows: List[List[String]] = (parsedJson \ "rows").extract[List[List[String]]]

    assert(header == List("X", "Y"))
    assert(rows == List(List("10", "20")))
  }

  def parseJsonToJValue(json: String): JValue = JsonMethods.parse(json)

}
