/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.partition

import com.johnsnowlabs.nlp.annotators.TextSplitter
import com.johnsnowlabs.reader.ElementType

import scala.collection.mutable
import scala.util.matching.Regex

private[johnsnowlabs] case class TitleChunkInput[A](
    source: A,
    text: String,
    metadata: Map[String, String],
    elementType: Option[String],
    pageNumber: Option[Int])

private[johnsnowlabs] case class TitleChunkSection[A](
    items: Vector[TitleChunkInput[A]],
    text: String)

private[johnsnowlabs] case class TitleChunkingOptions(
    joinString: String = " ",
    splitOnPageChange: Boolean = false,
    combineTextUnderNChars: Int = 0,
    enableOverflowSplitting: Boolean = true,
    maxCharacters: Int = 500,
    newAfterNChars: Int = -1,
    overlap: Int = 0,
    overlapAll: Boolean = false)

private[johnsnowlabs] object TitleChunkingUtil {

  def chunk[A](
      inputs: Seq[TitleChunkInput[A]],
      options: TitleChunkingOptions): Seq[TitleChunkSection[A]] = {
    require(options.maxCharacters > 0, "maxCharacters should be larger than 0.")
    require(options.combineTextUnderNChars >= 0, "combineTextUnderNChars should not be negative.")
    require(options.newAfterNChars >= -1, "newAfterNChars should be -1 or larger.")
    require(options.overlap >= 0, "overlap should not be negative.")
    if (options.enableOverflowSplitting) {
      require(
        options.overlap < options.maxCharacters,
        "overlap should be smaller than maxCharacters when overflow splitting is enabled.")
    }

    val sections = buildSections(inputs.filterNot(shouldDrop), options)
    val mergedSections = combineSmallSections(sections, options)

    val chunks = mutable.ListBuffer.empty[TitleChunkSection[A]]
    var lastNarrativeText = ""

    mergedSections.foreach { section =>
      val sectionText = joinTexts(section, options.joinString)
      if (sectionText.nonEmpty) {
        if (isTableSection(section)) {
          chunks += TitleChunkSection(section, sectionText)
          lastNarrativeText = ""
        } else {
          val content = withInterSectionOverlap(sectionText, lastNarrativeText, options)
          val splitChunks =
            if (options.enableOverflowSplitting && content.length > options.maxCharacters)
              splitOverflow(content, options)
            else Seq(content.trim)

          splitChunks.filter(_.nonEmpty).foreach { text =>
            chunks += TitleChunkSection(section, text)
          }
          lastNarrativeText = sectionText
        }
      }
    }

    chunks.toVector
  }

  private def buildSections[A](
      inputs: Seq[TitleChunkInput[A]],
      options: TitleChunkingOptions): Seq[Vector[TitleChunkInput[A]]] = {
    val sections = mutable.ListBuffer.empty[Vector[TitleChunkInput[A]]]
    var current = Vector.empty[TitleChunkInput[A]]
    var currentPage: Option[Int] = None

    def flushCurrent(): Unit = {
      if (current.nonEmpty) {
        sections += current
        current = Vector.empty
        currentPage = None
      }
    }

    inputs.foreach { input =>
      if (isTable(input)) {
        flushCurrent()
        sections += Vector(input)
      } else {
        val pageChanged =
          options.splitOnPageChange &&
            current.nonEmpty &&
            currentPage.isDefined &&
            input.pageNumber.isDefined &&
            currentPage != input.pageNumber

        val titleBoundary = current.nonEmpty && isTitle(input)
        val softLimitExceeded =
          options.newAfterNChars > 0 &&
            current.length >= 2 &&
            joinedLength(current :+ input, options.joinString) > options.newAfterNChars

        if (titleBoundary || pageChanged || softLimitExceeded) {
          flushCurrent()
        }

        current = current :+ input
        currentPage = input.pageNumber
      }
    }

    flushCurrent()
    sections.toVector
  }

  private def combineSmallSections[A](
      sections: Seq[Vector[TitleChunkInput[A]]],
      options: TitleChunkingOptions): Seq[Vector[TitleChunkInput[A]]] = {
    if (options.combineTextUnderNChars <= 0) {
      sections
    } else {
      sections.foldLeft(Vector.empty[Vector[TitleChunkInput[A]]]) { (acc, section) =>
        val canMerge =
          acc.nonEmpty &&
            !isTableSection(acc.last) &&
            !isTableSection(section) &&
            joinedLength(section, options.joinString) < options.combineTextUnderNChars

        if (canMerge) acc.init :+ (acc.last ++ section) else acc :+ section
      }
    }
  }

  private def splitOverflow(text: String, options: TitleChunkingOptions): Seq[String] = {
    val textSplitter = new TextSplitter(
      chunkSize = options.maxCharacters,
      chunkOverlap = options.overlap,
      keepSeparators = true,
      patternsAreRegex = true,
      trimWhitespace = true)

    val patterns =
      Seq(Regex.quote(options.joinString)).filter(_.nonEmpty).filterNot(_ == "\\Q\\E") ++ Seq(
        "\\s+")

    textSplitter
      .splitText(text, patterns.distinct)
      .map(_.trim)
      .filter(_.nonEmpty)
  }

  private def withInterSectionOverlap(
      sectionText: String,
      lastNarrativeText: String,
      options: TitleChunkingOptions): String = {
    val shouldPrefix =
      options.overlap > 0 &&
        lastNarrativeText.nonEmpty &&
        (options.overlapAll ||
          (options.enableOverflowSplitting && sectionText.length > options.maxCharacters))

    if (!shouldPrefix) {
      sectionText
    } else {
      val overlapPrefix = lastNarrativeText.takeRight(options.overlap).trim
      if (overlapPrefix.isEmpty) {
        sectionText
      } else if (options.joinString.isEmpty) {
        overlapPrefix + sectionText
      } else {
        overlapPrefix + options.joinString + sectionText
      }
    }
  }

  private def joinTexts[A](items: Seq[TitleChunkInput[A]], joinString: String): String = {
    items
      .flatMap(item => Option(item.text).map(_.trim).filter(_.nonEmpty))
      .mkString(joinString)
      .trim
  }

  private def joinedLength[A](items: Seq[TitleChunkInput[A]], joinString: String): Int =
    joinTexts(items, joinString).length

  private def isTableSection[A](items: Seq[TitleChunkInput[A]]): Boolean =
    items.nonEmpty && items.forall(isTable)

  private def isTable[A](input: TitleChunkInput[A]): Boolean =
    input.elementType.exists(_.equalsIgnoreCase(ElementType.TABLE))

  private def isTitle[A](input: TitleChunkInput[A]): Boolean =
    input.elementType.exists(_.equalsIgnoreCase(ElementType.TITLE))

  private def shouldDrop[A](input: TitleChunkInput[A]): Boolean = {
    !isTable(input) && Option(input.text).forall(_.trim.isEmpty)
  }
}
