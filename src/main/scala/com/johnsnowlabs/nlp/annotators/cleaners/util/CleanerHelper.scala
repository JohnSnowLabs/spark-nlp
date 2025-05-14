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
package com.johnsnowlabs.nlp.annotators.cleaners.util

import java.nio.charset.Charset
import java.util.regex.Pattern
import scala.util.matching.Regex

object CleanerHelper {

  private val UNICODE_BULLETS: List[String] = List(
    "\u0095",
    "\u2022",
    "\u2023",
    "\u2043",
    "\u3164",
    "\u204C",
    "\u204D",
    "\u2219",
    "\u25CB",
    "\u25CF",
    "\u25D8",
    "\u25E6",
    "\u2619",
    "\u2765",
    "\u2767",
    "\u29BE",
    "\u29BF",
    "\u002D",
    "",
    "\\*", // Escaped for regex compatibility
    "\u0095",
    "·")

  private val BULLETS_PATTERN = UNICODE_BULLETS.map(Pattern.quote).mkString("|")
  private val UNICODE_BULLETS_RE: Regex = new Regex(s"(?:$BULLETS_PATTERN)")

  private val HTML_APOSTROPHE_ENTITY: String = "&apos;"
  private val HEXADECIMAL_ESCAPE_SEQUENCE: Regex = """\\x([0-9A-Fa-f]{2})""".r

  /** Parses a string containing escape sequences (e.g., `\x9f`) into a byte array.
    *
    * @param text
    *   The input string with escape sequences.
    * @return
    *   A byte array representing the parsed bytes.
    */
  def parseEscapedBytes(text: String): Array[Byte] = {
    val RawByteCharset: Charset = Charset.forName("ISO-8859-1")

    // Replace escape sequences with their byte values
    HEXADECIMAL_ESCAPE_SEQUENCE
      .replaceAllIn(
        text,
        m => {
          val hexValue = m.group(1)
          Integer.parseInt(hexValue, 16).toChar.toString
        })
      .getBytes(RawByteCharset)
  }

  /** Formats an input encoding string (e.g., `utf-8`, `iso-8859-1`, etc).
    *
    * @param encoding
    *   The encoding string to be formatted.
    * @return
    *   The formatted encoding string.
    */
  def formatEncodingStr(encoding: String): String = {
    var formattedEncoding = encoding.toLowerCase.replace("_", "-")

    // Special case for Arabic and Hebrew charsets with directional annotations
    val annotatedEncodings = Set("iso-8859-6-i", "iso-8859-6-e", "iso-8859-8-i", "iso-8859-8-e")
    if (annotatedEncodings.contains(formattedEncoding)) {
      formattedEncoding = formattedEncoding.dropRight(2)
    }

    formattedEncoding
  }

  def cleanTrailingPunctuation(text: String): String = {
    text.replaceAll("[.,:;]+$", "")
  }

  def cleanDashes(text: String): String = {
    val dashRegex: Regex = "[-\u2013]".r
    dashRegex.replaceAllIn(text, " ").trim
  }

  def cleanExtraWhitespace(text: String): String = {
    // Replace all occurrences of '\xa0' (non-breaking space) with a regular space
    val hexNbspReplaced = text.replaceAll("\\\\x[aA]0", " ")

    // Normalize other whitespace characters if needed
    val normalizedText = hexNbspReplaced.replaceAll("\\p{Zs}", " ")

    // Collapse whitespace sequences into a single space
    val whitespaceRegex: Regex = "\\s+".r

    whitespaceRegex.replaceAllIn(normalizedText, " ").trim
  }

  def cleanBullets(text: String): String = {
    // Manually create a regex that explicitly matches the bullet "\u2022"
    val manualBulletRegex: Regex = new Regex(s"""^$UNICODE_BULLETS_RE\\s?""")

    // Debug the match
    manualBulletRegex.findPrefixOf(text) match {
      case Some(_) =>
        manualBulletRegex.replaceFirstIn(text, "").trim
      case None =>
        text
    }
  }

  def cleanNonAsciiChars(text: String): String = {
    val decodedText = HEXADECIMAL_ESCAPE_SEQUENCE.replaceAllIn(
      text,
      m => Integer.parseInt(m.group(1), 16).toChar.toString)

    val entityReplacedText = decodedText.replace(HTML_APOSTROPHE_ENTITY, "'")
    entityReplacedText.replaceAll("[^\u0020-\u007E]", "")
  }

  def cleanOrderedBullets(text: String): String = {
    val textParts = text.split("\\s+", 2) // Splitting into two parts to avoid unnecessary joins
    if (textParts.length < 2) return text

    val firstWord = textParts(0)
    val remainingText = textParts(1)

    if (!firstWord.contains(".") || firstWord.contains("..")) return text

    val bulletParts = firstWord.split("\\.")
    val cleanedBulletParts =
      if (bulletParts.last.isEmpty) bulletParts.dropRight(1) else bulletParts

    if (cleanedBulletParts.head.length > 2) text else remainingText.trim

  }

  def replaceUnicodeCharacters(text: String): String = {
    val decodedText = HEXADECIMAL_ESCAPE_SEQUENCE.replaceAllIn(
      text,
      m => {
        val hexValue = m.group(1)
        val byteValue = Integer.parseInt(hexValue, 16).toByte
        new String(Array(byteValue), Charset.forName("ISO-8859-1"))
      })

    val fullyDecodedText = new String(
      decodedText.getBytes(Charset.forName("ISO-8859-1")),
      Charset.forName("Windows-1252"))

    fullyDecodedText
      .replace("\u2018", "‘")
      .replace("\u2019", "’")
      .replace("\u201C", "“")
      .replace("\u201D", "”")
      .replace(HTML_APOSTROPHE_ENTITY, "'")
      .replace("â\u0080\u0099", "'")
      .replace("â\u0080“", "—")
      .replace("â\u0080”", "–")
      .replace("â\u0080¦", "…")
  }

  /** Removes punctuation from a given string.
    *
    * @params
    *   The input string.
    * @return
    *   The string with punctuation removed.
    */
  def removePunctuation(text: String): String = {
    // \p{P} matches any kind of punctuation character in Unicode
    val punctuationRegex = """\p{P}""".r
    punctuationRegex.replaceAllIn(text, "")
  }

  /** Cleans a prefix from a string based on a pattern.
    *
    * @param text
    *   The text to clean.
    * @return
    *   The cleaned string.
    */
  def cleanPrefix(text: String, pattern: String, ignoreCase: Boolean, strip: Boolean): String = {
    val regexStr =
      if (ignoreCase) s"(?i)^$pattern[\\p{Punct}\\s]*"
      else s"^$pattern[\\p{Punct}\\s]*"
    val regex = regexStr.r

    val cleanedText = regex.replaceAllIn(text, "")

    if (strip) cleanedText.replaceAll("^\\s+", "") else cleanedText
  }

  /** Cleans a postfix from a string based on a pattern.
    *
    * @param text
    *   The text to clean.
    * @return
    *   The cleaned string.
    */
  def cleanPostfix(text: String, pattern: String, ignoreCase: Boolean, strip: Boolean): String = {
    val regex = if (ignoreCase) s"(?i)$pattern$$".r else s"$pattern$$".r
    val cleanedText = regex.replaceAllIn(text, "")
    if (strip) cleanedText.trim else cleanedText
  }

  /** Converts a string representation of a byte string (e.g., containing escape sequences) to an
    * Annotation structure using the specified encoding.
    *
    * @param text
    *   The string representation of the byte string.
    * @return
    *   The String containing the decoded result
    */
  def bytesStringToString(text: String, encoding: String): String = {
    val textBytes = parseEscapedBytes(text)
    val formattedEncoding = formatEncodingStr(encoding)
    new String(textBytes, Charset.forName(formattedEncoding))
  }

}
