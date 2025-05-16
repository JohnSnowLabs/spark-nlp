package com.johnsnowlabs.reader.util

import com.johnsnowlabs.nlp.annotators.cleaners.util.CleanerHelper

import scala.util.matching.Regex

object TextParser {

  private val eBulletPattern: Regex = "^e$".r
  private val unicodeBulletsPattern: Regex =
    ("^[" + CleanerHelper.UNICODE_BULLETS.mkString("") + "]").r

  /** Groups paragraphs by processing text that uses blank lines to separate paragraphs.
    *
    * @param text
    *   The input text.
    * @param shortLineWordThreshold
    *   The maximum number of words a line can have to be considered "short". Lines with fewer
    *   words than this threshold will be treated as individual paragraphs. *
    * @return
    *   The processed text with paragraphs grouped.
    */
  def groupBrokenParagraphs(
      text: String,
      paragraphSplit: String,
      shortLineWordThreshold: Int): String = {
    // Split the text into paragraphs based on two or more newline sequences.
    val paragraphs: Array[String] = text.split(paragraphSplit)
    val cleanParagraphs = paragraphs.flatMap { paragraph =>
      if (paragraph.trim.isEmpty) {
        None
      } else {
        // Split the paragraph on single newline occurrences.
        val paraSplit: Array[String] = paragraph.split("""\s*\n\s*""")
        val allLinesShort =
          paraSplit.forall(line => line.trim.split("\\s+").length < shortLineWordThreshold)
        val trimmed = paragraph.trim
        if (unicodeBulletsPattern
            .findFirstIn(trimmed)
            .isDefined || eBulletPattern.findFirstIn(trimmed).isDefined) {
          groupBulletParagraph(paragraph)
        } else if (allLinesShort) {
          // If all lines are short, return the individual non-empty lines.
          paraSplit.filter(_.trim.nonEmpty).toSeq
        } else {
          // Otherwise, replace newline sequences within the paragraph with a space.
          Seq(paragraph.replaceAll("""\s*\n\s*""", " "))
        }
      }
    }
    cleanParagraphs.mkString("\n\n")
  }

  private def groupBulletParagraph(paragraph: String): Seq[String] = {
    paragraph.split("\n").map(_.trim).filter(_.nonEmpty).toSeq
  }

  /** autoParagraphGrouper determines which paragraph grouping method to use based on the ratio of
    * empty lines.
    *
    * @param text
    *   The input text.
    * @param maxLineCount
    *   Maximum number of lines to inspect from the text when calculating the empty line ratio.
    * @param threshold
    *   The ratio threshold (empty lines / total lines) to decide which grouper to use. If the
    *   ratio is below this value, newLineGrouper is used; otherwise, groupBrokenParagraphs is
    *   used.
    * @return
    *   The processed text.
    */
  def autoParagraphGrouper(
      text: String,
      paragraphSplit: String,
      maxLineCount: Int,
      threshold: Double,
      shortLineWordThreshold: Int): String = {
    val lines = text.split("\n")
    val count = Math.min(lines.length, maxLineCount)
    var emptyLineCount = 0
    for (i <- 0 until count) {
      if (lines(i).trim.isEmpty) emptyLineCount += 1
    }
    val ratio = emptyLineCount.toDouble / count
    if (ratio < threshold) newLineGrouper(text)
    else groupBrokenParagraphs(text, paragraphSplit, shortLineWordThreshold)
  }

  // newLineGrouper concatenates text that uses a one-line paragraph break pattern.
  private def newLineGrouper(text: String): String = {
    val paragraphs = text.split("\n").map(_.trim).filter(_.nonEmpty)
    paragraphs.mkString("\n\n")
  }

}
