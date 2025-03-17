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
package com.johnsnowlabs.nlp.annotators.cleaners

import com.johnsnowlabs.ml.tensorflow.sentencepiece.ReadSentencePieceModel
import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import com.johnsnowlabs.nlp.annotators.cleaners.util.CleanerHelper
import com.johnsnowlabs.nlp.annotators.cleaners.util.CleanerHelper._
import com.johnsnowlabs.nlp.annotators.seq2seq.{
  MarianTransformer,
  ReadMarianMTDLModel,
  ReadablePretrainedMarianMTModel
}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

class Cleaner(override val uid: String) extends MarianTransformer {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("CLEANER"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  val encoding = new Param[String](
    this,
    "encoding",
    "The encoding to be used for decoding the byte string (default is utf-8)")

  def setEncoding(value: String): this.type = set(this.encoding, value)

  val cleanPrefixPattern = new Param[String](
    this,
    "cleanPrefixPattern",
    "The pattern for the prefix. Can be a simple string or a regex pattern.")

  def setCleanPrefixPattern(value: String): this.type = set(this.cleanPrefixPattern, value)

  val cleanPostfixPattern = new Param[String](
    this,
    "cleanPostfixPattern",
    "The pattern for the postfix. Can be a simple string or a regex pattern.")

  def setCleanPostfixPattern(value: String): this.type = set(this.cleanPrefixPattern, value)

  /** cleanerMode can take the following values:
    *   - `bytes_string_to_string`: Converts a string representation of a byte string (e.g.,
    *     containing escape sequences) to an Annotation structure using the specified encoding.
    */
  val cleanerMode: Param[String] = new Param[String](
    this,
    "cleanerMode",
    "possible values: " +
      "clean, bytes_string_to_string, clean_non_ascii_chars, clean_ordered_bullets, clean_postfix," +
      " clean_prefix, remove_punctuation, replace_unicode_characters")

  def setCleanerMode(value: String): this.type = {
    value.trim.toLowerCase() match {
      case "clean" => set(this.cleanerMode, value)
      case "bytes_string_to_string" => set(this.cleanerMode, value)
      case "clean_non_ascii_chars" => set(this.cleanerMode, value)
      case "clean_ordered_bullets" => set(this.cleanerMode, value)
      case "clean_postfix" => set(this.cleanerMode, value)
      case "clean_prefix" => set(this.cleanerMode, value)
      case "remove_punctuation" => set(this.cleanerMode, value)
      case "replace_unicode_characters" => set(this.cleanerMode, value)
      case "translate" => set(this.cleanerMode, value)
      case _ => throw new IllegalArgumentException(s"Cleaner mode $value is not supported.")
    }
    set(this.cleanerMode, value)
  }

  val extraWhitespace =
    new Param[Boolean](this, "extraWhitespace", "Whether to remove extra whitespace.")

  def setExtraWhitespace(value: Boolean): this.type = set(this.extraWhitespace, value)

  val dashes = new Param[Boolean](this, "dashes", "Whether to handle dashes in text.")

  def setDashes(value: Boolean): this.type = set(this.dashes, value)

  val bullets = new Param[Boolean](this, "bullets", "Whether to handle bullets in text.")

  def setBullets(value: Boolean): this.type = set(this.bullets, value)

  val trailingPunctuation = new Param[Boolean](
    this,
    "trailingPunctuation",
    "Whether to remove trailing punctuation from text.")

  def setTrailingPunctuation(value: Boolean): this.type = set(this.trailingPunctuation, value)

  val lowercase = new Param[Boolean](this, "lowercase", "Whether to convert text to lowercase.")

  def setLowercase(value: Boolean): this.type = set(this.lowercase, value)

  val ignoreCase = new Param[Boolean](this, "ignoreCase", "If true, ignores case in the pattern.")

  def setIgnoreCase(value: Boolean): this.type = set(this.ignoreCase, value)

  val strip = new Param[Boolean](
    this,
    "strip",
    "If true, removes leading or trailing whitespace from the cleaned string.")

  def setStrip(value: Boolean): this.type = set(this.strip, value)

  setDefault(
    encoding -> "utf-8",
    extraWhitespace -> false,
    dashes -> false,
    bullets -> false,
    trailingPunctuation -> false,
    lowercase -> false,
    ignoreCase -> false,
    strip -> true,
    cleanerMode -> "translate")

  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    require($(cleanerMode) != "undefined", "Extractor mode must be set.")

    if ($(cleanerMode) == "translate") {
      return super.batchAnnotate(batchedAnnotations)
    }

    batchedAnnotations.map { annotations =>
      $(cleanerMode) match {
        case "clean" => annotations.map(buildAnnotation(clean)).toSeq
        case "bytes_string_to_string" =>
          annotations.map(buildAnnotation(bytesStringToString)).toSeq
        case "clean_non_ascii_chars" => annotations.map(buildAnnotation(cleanNonAsciiChars)).toSeq
        case "clean_ordered_bullets" =>
          annotations.map(buildAnnotation(cleanOrderedBullets)).toSeq
        case "clean_postfix" => annotations.map(buildAnnotation(cleanPostfix)).toSeq
        case "clean_prefix" => annotations.map(buildAnnotation(cleanPrefix)).toSeq
        case "remove_punctuation" => annotations.map(buildAnnotation(removePunctuation)).toSeq
        case "replace_unicode_characters" =>
          annotations.map(buildAnnotation(replaceUnicodeCharacters)).toSeq
      }
    }
  }

  def buildAnnotation(transformation: String => String)(annotation: Annotation): Annotation = {
    val cleanText = transformation(annotation.result)
    Annotation(
      annotatorType = outputAnnotatorType,
      begin = 0,
      end = cleanText.length,
      result = cleanText,
      metadata = Map())
  }

  /** Converts a string representation of a byte string (e.g., containing escape sequences) to an
    * Annotation structure using the specified encoding.
    *
    * @param text
    *   The string representation of the byte string.
    * @return
    *   The String containing the decoded result
    */
  private def bytesStringToString(text: String): String = {
    CleanerHelper.bytesStringToString(text, $(encoding))
  }

  private def clean(text: String): String = {

    var cleanedText = if ($(lowercase)) text.toLowerCase else text
    cleanedText =
      if ($(trailingPunctuation)) cleanTrailingPunctuation(cleanedText) else cleanedText
    cleanedText = if ($(dashes)) cleanDashes(cleanedText) else cleanedText
    cleanedText = if ($(extraWhitespace)) cleanExtraWhitespace(cleanedText) else cleanedText
    cleanedText = if ($(bullets)) cleanBullets(cleanedText) else cleanedText

    cleanedText.trim
  }

  /** Cleans a prefix from a string based on a pattern.
    *
    * @param text
    *   The text to clean.
    * @return
    *   The cleaned string.
    */
  private def cleanPrefix(text: String): String = {
    CleanerHelper.cleanPrefix(text, $(cleanPrefixPattern), $(ignoreCase), $(strip))
  }

  /** Cleans a postfix from a string based on a pattern.
    *
    * @param text
    *   The text to clean.
    * @return
    *   The cleaned string.
    */
  private def cleanPostfix(text: String): String = {
    CleanerHelper.cleanPostfix(text, $(cleanPrefixPattern), $(ignoreCase), $(strip))
  }

}

object Cleaner
    extends ReadablePretrainedMarianMTModel
    with ReadMarianMTDLModel
    with ReadSentencePieceModel
