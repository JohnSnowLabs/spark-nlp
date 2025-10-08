/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.reader.util

import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.jsoup.Jsoup
import org.jsoup.nodes.Element

import scala.jdk.CollectionConverters.asScalaBufferConverter

object HTMLParser {

  private val PtToPx = 1.333 // 1pt ≈ 1.333px (CSS standard conversion)

  /** Extracts font size from a CSS style string, supporting 'px', 'pt', 'em', 'rem', and '%'.
    * Returns pixel size as integer.
    *
    * @param style
    *   CSS style string (e.g., "font-size: 1.2em; ...")
    * @param baseEmPx
    *   Base pixel size for 1em (default 16)
    * @param baseRemPx
    *   Base pixel size for 1rem (default 16)
    * @param parentPx
    *   Parent font size for '%' (default 16)
    * @return
    *   Font size in pixels, or 0 if not found
    */
  def extractFontSize(
      style: String,
      baseEmPx: Int = 16,
      baseRemPx: Int = 16,
      parentPx: Int = 16): Int = {
    val sizePattern = """(?i)font-size\s*:\s*([0-9.]+)\s*(px|pt|em|rem|%)""".r
    sizePattern.findFirstMatchIn(style) match {
      case Some(m) =>
        val value = m.group(1).toDouble
        m.group(2).toLowerCase match {
          case "px" => Math.round(value).toInt
          case "pt" => Math.round(value * PtToPx).toInt
          case "em" => Math.round(value * baseEmPx).toInt
          case "rem" => Math.round(value * baseRemPx).toInt
          case "%" => Math.round(parentPx * value / 100).toInt
          case _ => 0
        }
      case None => 0
    }
  }

  def isFormattedAsTitle(style: String, titleFontSize: Int): Boolean = {
    val lowerStyle = style.toLowerCase

    // Matches 'font-weight:bold', 'font-weight:bolder', 'font-weight:700', 'font-weight:800', 'font-weight:900'
    val boldPattern = """font-weight\s*:\s*(bold(er)?|[7-9]00)\b""".r
    val isBold = boldPattern.findFirstIn(lowerStyle).isDefined

    val isLargeFont =
      lowerStyle.contains("font-size") && extractFontSize(lowerStyle) >= titleFontSize
    val isCentered = lowerStyle.contains("text-align:center")

    isBold || isLargeFont || (isCentered && isBold) || (isCentered && isLargeFont)
  }

  def isTitleElement(tag: String, style: String, role: String, titleFontSize: Int): Boolean = {
    // Recognize titles from common title-related tags or formatted <p> elements
    tag match {
      case "title" | "h1" | "h2" | "h3" | "header" => true
      case "p" | "div" => isFormattedAsTitle(style, titleFontSize)
      case _ => role == "heading" // ARIA role="heading"
    }
  }

  def tableElementToJson(tableElem: Element): String = {
    implicit val formats = Serialization.formats(NoTypeHints)

    val caption = Option(tableElem.selectFirst("caption"))
      .map(_.text.trim)
      .getOrElse("")

    val allRows = tableElem.select("tr").asScala.toList

    val headerRowOpt = allRows.find(tr => tr.select("th").asScala.nonEmpty)

    val headers: List[String] = headerRowOpt
      .map(_.select("th,td").asScala.map(_.text.trim).toList)
      .getOrElse(List.empty)

    val headerIndexOpt = headerRowOpt.map(allRows.indexOf)

    val dataRows =
      allRows.zipWithIndex
        .filter { case (_, idx) => !headerIndexOpt.contains(idx) }
        .map(_._1)
        .map(row => row.select("td").asScala.map(_.text.trim).toList)
        .filter(_.nonEmpty)

    val jsonObj = Map("caption" -> caption, "header" -> headers, "rows" -> dataRows)

    Serialization.write(jsonObj)
  }

  def parseFirstTableElement(html: String): Element = Jsoup.parse(html).select("table").first()

}
