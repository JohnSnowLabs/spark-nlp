package com.johnsnowlabs.nlp.annotators
import scala.collection.mutable

/** Splits texts recursively to match given length
  *
  * @param chunkSize
  *   Length of the text chunks, measured by `lengthFunction`
  * @param chunkOverlap
  *   Overlap of the text chunks
  * @param keepSeparators
  *   Whether to keep separators in the final chunks
  * @param patternsAreRegex
  *   Whether to interpret split patterns as regex
  * @param trimWhitespace
  *   Whether to trim the whitespace from the final chunks
  * @param lengthFunction
  *   Function to measure chunk length
  */
class TextSplitter(
    chunkSize: Int,
    chunkOverlap: Int,
    keepSeparators: Boolean,
    patternsAreRegex: Boolean,
    trimWhitespace: Boolean,
    lengthFunction: String => Int = _.length) {

  def joinDocs(currentDoc: Seq[String], separator: String): String = {
    val joinSeparator = if (patternsAreRegex && !keepSeparators) "" else separator
    val joined = String.join(joinSeparator, currentDoc: _*)

    if (trimWhitespace) joined.trim else joined
  }

  /** Splits the given text with the separator.
    *
    * The separator is assumed to be regex (which was optionally escaped).
    *
    * @param text
    *   Text to split
    * @param separator
    *   Regex as String
    * @return
    */
  def splitTextWithRegex(text: String, separator: String): Seq[String] = {
    val splits: Seq[String] = if (separator.nonEmpty) {
      val pattern = if (keepSeparators) f"(?=$separator)" else separator
      text.split(pattern)
    } else Seq(text)

    splits.filter(_.nonEmpty)
  }

  /** Combines smaller text chunks into one that has about the size of chunk size.
    *
    * @param splits
    *   Splits from the previous separator
    * @param separator
    *   The current separator
    * @return
    */
  def mergeSplits(splits: Seq[String], separator: String): Seq[String] = {
    val separatorLen = lengthFunction(separator)

    var docs: mutable.Seq[String] = mutable.Seq()
    var currentDoc: mutable.Seq[String] = mutable.Seq()
    var total: Int = 0

    splits.foreach { d =>
      val len = lengthFunction(d)

      def separatorLenNonEmpty = if (currentDoc.nonEmpty) separatorLen else 0

      def separatorLenActualText =
        if (currentDoc.length > 1) separatorLen
        else 0

      if (total + len + separatorLenNonEmpty > chunkSize) {
        if (currentDoc.nonEmpty) {
          val doc = joinDocs(currentDoc, separator)
          if (doc.nonEmpty) {
            docs = docs :+ doc
          }

          def mergeLargerThanChunkSize =
            total + len + separatorLenNonEmpty > chunkSize && total > 0

          while (total > chunkOverlap || mergeLargerThanChunkSize) {
            total -= lengthFunction(currentDoc.head) + separatorLenActualText
            currentDoc = currentDoc.drop(1)
          }
        }
      }

      currentDoc = currentDoc :+ d
      total += len + separatorLenActualText
    }

    val doc = joinDocs(currentDoc, separator)
    if (doc.nonEmpty) {
      docs = docs :+ doc
    }

    docs
  }

  // noinspection RegExpRedundantEscape
  def escapeRegexIfNeeded(text: String) =
    if (patternsAreRegex) text
    else text.replaceAll("([\\\\\\.\\[\\{\\(\\*\\+\\?\\^\\$\\|])", "\\\\$1")

  /** Splits a text into chunks of roughly given chunk size. The separators are given in a list
    * and will be used in order.
    *
    * Inspired by LangChain's RecursiveCharacterTextSplitter.
    *
    * @param text
    *   Text to split
    * @param separators
    *   List of separators in decreasing priority
    * @return
    */
  def splitText(text: String, separators: Seq[String]): Seq[String] = {
    // Get appropriate separator to use
    val (separator: String, nextSeparators: Seq[String]) = separators
      .map(escapeRegexIfNeeded)
      .zipWithIndex
      .collectFirst {
        case (sep, _) if sep.isEmpty =>
          (sep, Seq.empty)
        case (sep, i) if sep.r.findFirstIn(text).isDefined =>
          (sep, separators.drop(i + 1))
      }
      .getOrElse(("", Seq.empty))

    val splits = splitTextWithRegex(text, separator)

    // Now go merging things, recursively splitting longer texts.
    var finalChunks: mutable.Seq[String] = mutable.Seq()
    var goodSplits: mutable.Seq[String] = mutable.Seq.empty
    val separatorStr = if (keepSeparators) "" else separator

    splits.foreach { s =>
      if (lengthFunction(s) < chunkSize) {
        goodSplits = goodSplits :+ s
      } else {
        if (goodSplits.nonEmpty) {
          val mergedText = mergeSplits(goodSplits, separatorStr)
          finalChunks = finalChunks ++ mergedText
          goodSplits = mutable.Seq.empty
        }
        if (nextSeparators.isEmpty) {
          finalChunks = finalChunks :+ s
        } else {
          val recursiveChunks = splitText(s, nextSeparators)
          finalChunks = finalChunks ++ recursiveChunks
        }
      }
    }

    if (goodSplits.nonEmpty) {
      val mergedText = mergeSplits(goodSplits, separatorStr)
      finalChunks = finalChunks ++ mergedText
    }

    finalChunks
  }
}
