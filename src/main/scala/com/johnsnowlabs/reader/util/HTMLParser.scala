package com.johnsnowlabs.reader.util

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

}
