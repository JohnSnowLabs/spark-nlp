/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.annotators.cleaners.util.CleanerHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasSimpleAnnotate}
import org.apache.spark.ml.param.{BooleanParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.slf4j.{Logger, LoggerFactory}

import java.nio.charset.{Charset, StandardCharsets}
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex
import scala.util.{Failure, Success, Try}
import scala.xml.XML

/** Annotator which normalizes raw text from tagged text, e.g. scraped web pages or xml documents,
  * from document type columns into Sentence. Removes all dirty characters from text following one
  * or more input regex patterns. Can apply not wanted character removal with a specific policy.
  * Can apply lower case normalization.
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-normalizer/document_normalizer_notebook.ipynb Examples]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.DocumentNormalizer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val cleanUpPatterns = Array("<[^>]*>")
  *
  * val documentNormalizer = new DocumentNormalizer()
  *   .setInputCols("document")
  *   .setOutputCol("normalizedDocument")
  *   .setAction("clean")
  *   .setPatterns(cleanUpPatterns)
  *   .setReplacement(" ")
  *   .setPolicy("pretty_all")
  *   .setLowercase(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   documentNormalizer
  * ))
  *
  * val text =
  *   """
  * <div id="theworldsgreatest" class='my-right my-hide-small my-wide toptext' style="font-family:'Segoe UI',Arial,sans-serif">
  *   THE WORLD'S LARGEST WEB DEVELOPER SITE
  *   <h1 style="font-size:300%;">THE WORLD'S LARGEST WEB DEVELOPER SITE</h1>
  *   <p style="font-size:160%;">Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum..</p>
  * </div>
  *
  * </div>"""
  * val data = Seq(text).toDF("text")
  * val pipelineModel = pipeline.fit(data)
  *
  * val result = pipelineModel.transform(data)
  * result.selectExpr("normalizedDocument.result").show(truncate=false)
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[ the world's largest web developer site the world's largest web developer site lorem ipsum is simply dummy text of the printing and typesetting industry. lorem ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. it has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. it was popularised in the 1960s with the release of letraset sheets containing lorem ipsum passages, and more recently with desktop publishing software like aldus pagemaker including versions of lorem ipsum..]|
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * }}}
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class DocumentNormalizer(override val uid: String)
    extends AnnotatorModel[DocumentNormalizer]
    with HasSimpleAnnotate[DocumentNormalizer]{

  private val logger: Logger = LoggerFactory.getLogger(this.getClass)

  private val EMPTY_STR = ""
  private val BREAK_STR = "|##|"
  private val SPACE_STR = " "
  private val GENERIC_TAGS_REMOVAL_PATTERN = "<[^>]*>"

  /** Input annotator type : DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  /** Output annotator type : DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  def this() = this(Identifiable.randomUID("DOCUMENT_NORMALIZER"))

  /** Action to perform applying regex patterns on text
    *
    * @group param
    */
  val action: Param[String] =
    new Param(this, "action", "Action to perform applying regex patterns on text")

  /** Normalization regex patterns which match will be removed from document (Default:
    * `Array("<[^>]*>")`)
    *
    * @group param
    */
  val patterns: StringArrayParam = new StringArrayParam(
    this,
    "patterns",
    "Normalization regex patterns which match will be removed from document. Defaults is \"<[^>]*>\"")

  /** Replacement string to apply when regexes match (Default: `" "`)
    *
    * @group param
    */
  val replacement: Param[String] =
    new Param(this, "replacement", "Replacement string to apply when regexes match")

  /** Whether to convert strings to lowercase (Default: `false`)
    *
    * @group param
    */
  val lowercase = new BooleanParam(
    this,
    "lowercase",
    "Whether to convert strings to lowercase (Default: `false`)")

  /** RemovalPolicy to remove patterns from text with a given policy (Default: `"pretty_all"`).
    * Possible values are `"all", "pretty_all", "first", "pretty_first"`
    * @group param
    */
  val policy: Param[String] =
    new Param(this, "policy", "RemovalPolicy to remove pattern from text")

  /** File encoding to apply on normalized documents (Default: `"disable"`)
    *
    * @group param
    */
  val encoding: Param[String] = new Param(
    this,
    name = "encoding",
    "File encoding to apply on normalized documents (Default: `disable`)")

  val presetPattern = new Param[String](
    this,
    "presetPattern",
    "Single functional cleaner preset (CLEAN_BULLETS, CLEAN_DASHES, etc.)"
  )

  val autoMode = new Param[String](this, "autoMode",
    "Automatic cleaning mode grouping multiple cleaners: light_clean, document_clean, social_clean, html_clean, full_auto")

  //  Assuming non-html does not contain any < or > and that input string is correctly structured
  setDefault(
    inputCols -> Array(AnnotatorType.DOCUMENT),
    action -> "clean",
    patterns -> Array(GENERIC_TAGS_REMOVAL_PATTERN),
    replacement -> SPACE_STR,
    lowercase -> false,
    policy -> "pretty_all",
    encoding -> "disable")

  /** Action to perform on text. (Default `"clean"`).
    *
    * @group getParam
    */
  def getAction: String = $(action)

  /** Regular expressions list for normalization.
    *
    * @group getParam
    */
  def getPatterns: Array[String] = $(patterns)

  /** Replacement string to apply when regexes match (Default: `" "`)
    *
    * @group getParam
    */
  def getReplacement: String = $(replacement)

  /** Lowercase tokens (Default: `false`)
    *
    * @group getParam
    */
  def getLowercase: Boolean = $(lowercase)

  /** Policy to remove patterns from text (Default: `"pretty_all"`)
    *
    * @group getParam
    */
  def getPolicy: String = $(policy)

  /** Encoding to apply to normalized documents (Default: `"disable"`)
    *
    * @group getParam
    */
  def getEncoding: String = $(encoding)

  /** Action to perform on text. (Default `"clean"`).
    *
    * @group getParam
    */
  def setAction(value: String): this.type = set(action, value)

  /** Regular expressions list for normalization (Default: `Array("<[^>]*>")`)
    *
    * @group setParam
    */
  def setPatterns(value: Array[String]): this.type = set(patterns, value)

  /** Replacement string to apply when regexes match (Default: `" "`)
    *
    * @group getParam
    */
  def setReplacement(value: String): this.type = set(replacement, value)

  /** Lower case tokens default false
    *
    * @group setParam
    */
  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  /** Removal policy to apply (Default: `"pretty_all"`). Valid policy values are: "all",
    * "pretty_all", "first", "pretty_first"
    *
    * @group setParam
    */
  def setPolicy(value: String): this.type = set(policy, value)

  /** Encoding to apply. Default is `"UTF-8"`. Valid encoding are values are: UTF_8, UTF_16,
    * US_ASCII, ISO-8859-1, UTF-16BE, UTF-16LE
    *
    * @group setParam
    */
  def setEncoding(value: String): this.type = set(encoding, value)

  def setPresetPattern(value: String): this.type = set(presetPattern, value)

  /** Applying document normalization without pretty formatting (removing multiple spaces) */
  private def withAllFormatter(
      text: String,
      action: String,
      patterns: Array[String],
      replacement: String): String = {
    action match {
      case "clean" =>
        val patternsStr: String = patterns.mkString(BREAK_STR)
        text.replaceAll(patternsStr, replacement)
      case "extract" =>
        val htmlXml = XML.loadString(text)
        val textareaContents = (htmlXml \\ patterns.mkString).text
        textareaContents
      case "lookaround" =>
        LookAroundManager.process(text, patterns, replacement)
      case _ =>
        throw new Exception(
          "Unknown action parameter in DocumentNormalizer annotation." +
            "Please select either: clean or extract")
    }
  }

  /** pattern to grab from text as token candidates. Defaults \\S+ */
  private def withPrettyAllFormatter(
      text: String,
      action: String,
      patterns: Array[String],
      replacement: String): String = {
    withAllFormatter(text, action, patterns, replacement)
      .split("\\s+")
      .map(_.trim)
      .mkString(SPACE_STR)
  }

  /** Applying document normalization without pretty formatting (removing multiple spaces)
    * retrieving first element only
    */
  private def withFirstFormatter(
      text: String,
      action: String,
      patterns: Array[String],
      replacement: String): String = {
    action match {
      case "clean" =>
        val patternsStr: String = patterns.mkString(BREAK_STR)
        text.replaceFirst(patternsStr, replacement)
      case "extract" =>
        val htmlXml = XML.loadString(text)
        val textareaContents = htmlXml \\ patterns.mkString
        textareaContents.head.mkString
      case "lookaround" =>
        LookAroundManager.process(text, patterns, replacement)
      case _ =>
        throw new Exception(
          "Unknown action parameter in DocumentNormalizer annotation." +
            "Please select either: clean or extract")
    }
  }

  /** pattern to grab from text as token candidates. Defaults \\S+ */
  private def withPrettyFirstFormatter(
      text: String,
      action: String,
      patterns: Array[String],
      replacement: String): String = {
    withFirstFormatter(text, action, patterns, replacement)
      .split("\\s+")
      .map(_.trim)
      .mkString(SPACE_STR)
  }

  /** Apply a given encoding to the processed text
    *
    * US-ASCII Seven-bit ASCII, a.k.a. ISO646-US, a.k.a. the Basic Latin block of the Unicode
    * character set
    *
    * ISO-8859-1 ISO Latin Alphabet No. 1, a.k.a. ISO-LATIN-1
    *
    * UTF-8 Eight-bit UCS Transformation Format
    *
    * UTF-16BE Sixteen-bit UCS Transformation Format, big-endian byte order
    *
    * UTF-16LE Sixteen-bit UCS Transformation Format, little-endian byte order
    *
    * UTF-16 Sixteen-bit UCS Transformation Format, byte order identified by an optional
    * byte-order mark
    */
  private def withEncoding(text: String, encoding: Charset = StandardCharsets.UTF_8): String = {
    val defaultCharset: Charset = Charset.defaultCharset
    if (!Charset.defaultCharset.equals(encoding)) {
      log.warn("Requested encoding parameter is different from the default charset.")
    }
    new String(text.getBytes(defaultCharset), encoding)
  }

  /** Apply document normalization on text using action, patterns, policy, lowercase and encoding
    * parameters.
    */
//  private def applyDocumentNormalization(
//      text: String,
//      action: String,
//      patterns: Array[String],
//      replacement: String,
//      policy: String,
//      lowercase: Boolean,
//      encoding: String): String = {
//
//    val builtinPatterns: Array[String] = {
//      if (isDefined(presetPattern)) {
//        val presetName = $(presetPattern)
//        BUILTIN_PATTERNS.getOrElse(
//          presetName,
//          throw new IllegalArgumentException(
//            s"Unknown preset pattern '$presetName'. Valid presets: ${BUILTIN_PATTERNS.keys.mkString(", ")}"
//          )
//        )
//      } else {
//        Array.empty[String]
//      }
//    }
//
//    val combinedPatterns: Array[String] = {
//      val userPatterns = Option(patterns).getOrElse(Array.empty[String])
//      (builtinPatterns ++ userPatterns).distinct
//    }
//
//    val processedWithActionPatterns: String = policy match {
//      case "all" => withAllFormatter(text, action, combinedPatterns, replacement)
//      case "pretty_all" => withPrettyAllFormatter(text, action, combinedPatterns, replacement)
//      case "first" => withFirstFormatter(text, action, combinedPatterns, replacement)
//      case "pretty_first" => withPrettyFirstFormatter(text, action, combinedPatterns, replacement)
//      case _ =>
//        throw new Exception(
//          "Unknown policy parameter in DocumentNormalizer annotation." +
//            "Please select either: all, pretty_all, first, or pretty_first")
//    }
//
//    val cased =
//      if (lowercase)
//        processedWithActionPatterns.toLowerCase
//      else
//        processedWithActionPatterns
//
//    encoding match {
//      case "disable" => cased
//      case "UTF-8" => withEncoding(cased, StandardCharsets.UTF_8)
//      case "UTF-16" => withEncoding(cased, StandardCharsets.UTF_16)
//      case "US-ASCII" => withEncoding(cased, StandardCharsets.US_ASCII)
//      case "ISO-8859-1" => withEncoding(cased, StandardCharsets.ISO_8859_1)
//      case "UTF-16BE" => withEncoding(cased, StandardCharsets.UTF_16BE)
//      case "UTF-16LE" => withEncoding(cased, StandardCharsets.UTF_16LE)
//      case _ =>
//        throw new Exception("Unknown encoding parameter in DocumentNormalizer annotation." +
//          "Please select either: disable, UTF_8, UTF_16, US_ASCII, ISO-8859-1, UTF-16BE, UTF-16LE")
//    }
//  }

  /**
   * Applies document normalization:
   *   1. User-defined regex patterns (if any)
   *   2. Either functional preset (CleanerHelper) OR autoMode (CleanerHelper bundles)
   *   3. Lowercasing and encoding policy
   */
  private def applyDocumentNormalization(
                                          text: String,
                                          action: String,
                                          patterns: Array[String],
                                          replacement: String,
                                          policy: String,
                                          lowercase: Boolean,
                                          encoding: String
                                        ): String = {

    val hasPreset = isDefined(presetPattern) && FUNCTIONAL_PRESETS.contains($(presetPattern))
    val hasAutoMode = isDefined(autoMode) && AUTO_MODE_FUNCTIONS.contains($(autoMode))

    val selectedCleaner: Either[String => String, Seq[String => String]] = (hasPreset, hasAutoMode) match {
      case (true, true) =>
        logger.warn(
          s"[DocumentNormalizer] Both presetPattern (${ $(presetPattern) }) and autoMode (${ $(autoMode) }) are set. " +
            s"autoMode will take precedence."
        )
        Right(AUTO_MODE_FUNCTIONS($(autoMode)))
      case (true, false) =>
        Left(FUNCTIONAL_PRESETS($(presetPattern)))
      case (false, true) =>
        Right(AUTO_MODE_FUNCTIONS($(autoMode)))
      case _ =>
        Right(Seq.empty)
    }

    val userPatterns = Option(patterns).getOrElse(Array.empty[String])

    val regexCleanedText: String = if (userPatterns.nonEmpty) {
      policy match {
        case "all"          => withAllFormatter(text, action, userPatterns, replacement)
        case "pretty_all"   => withPrettyAllFormatter(text, action, userPatterns, replacement)
        case "first"        => withFirstFormatter(text, action, userPatterns, replacement)
        case "pretty_first" => withPrettyFirstFormatter(text, action, userPatterns, replacement)
        case _ =>
          throw new Exception(
            "Unknown policy parameter in DocumentNormalizer. " +
              "Valid options: all, pretty_all, first, pretty_first."
          )
      }
    } else text

    val cleanedText: String = selectedCleaner match {
      case Left(fn) =>
        logger.info(s"[DocumentNormalizer] Applying preset cleaner: ${ $(presetPattern) }")
        fn(regexCleanedText)

      case Right(funcs) if funcs.nonEmpty =>
        val modeName = $(autoMode)
        val functionNames = funcs.flatMap(FUNCTION_NAME_LOOKUP.get)
        logger.info(
          s"[DocumentNormalizer] AutoMode '$modeName' active. Applying cleaners in order: " +
            functionNames.mkString(", ")
        )
        funcs.foldLeft(regexCleanedText) { (acc, cleanerFn) => cleanerFn(acc) }

      case _ =>
        regexCleanedText
    }

    val casedText = if (lowercase) cleanedText.toLowerCase else cleanedText

    encoding match {
      case "disable"    => casedText
      case "UTF-8"      => withEncoding(casedText, StandardCharsets.UTF_8)
      case "UTF-16"     => withEncoding(casedText, StandardCharsets.UTF_16)
      case "US-ASCII"   => withEncoding(casedText, StandardCharsets.US_ASCII)
      case "ISO-8859-1" => withEncoding(casedText, StandardCharsets.ISO_8859_1)
      case "UTF-16BE"   => withEncoding(casedText, StandardCharsets.UTF_16BE)
      case "UTF-16LE"   => withEncoding(casedText, StandardCharsets.UTF_16LE)
      case other =>
        throw new Exception(
          s"Unknown encoding parameter: $other. " +
            "Please select one of disable, UTF-8, UTF-16, US-ASCII, ISO-8859-1, UTF-16BE, UTF-16LE."
        )
    }
  }

  private lazy val FUNCTIONAL_PRESETS: Map[String, String => String] = Map(
    "CLEAN_BULLETS" -> CleanerHelper.cleanBullets,
    "CLEAN_ORDERED_BULLETS" -> CleanerHelper.cleanOrderedBullets,
    "CLEAN_DASHES" -> CleanerHelper.cleanDashes,
    "CLEAN_TRAILING_PUNCTUATION" -> CleanerHelper.cleanTrailingPunctuation,
    "CLEAN_EXTRA_WHITESPACE" -> CleanerHelper.cleanExtraWhitespace,
    "REMOVE_PUNCTUATION" -> CleanerHelper.removePunctuation,
    "CLEAN_NON_ASCII" -> CleanerHelper.cleanNonAsciiChars,
    "REPLACE_UNICODE" -> CleanerHelper.replaceUnicodeCharacters
  )

  private lazy val AUTO_MODE_FUNCTIONS: Map[String, Seq[String => String]] = Map(
    "light_clean" -> Seq(
      CleanerHelper.cleanExtraWhitespace,
      CleanerHelper.cleanTrailingPunctuation
    ),
    "document_clean" -> Seq(
      CleanerHelper.cleanBullets,
      CleanerHelper.cleanOrderedBullets,
      CleanerHelper.cleanDashes,
      CleanerHelper.cleanExtraWhitespace
    ),
    "social_clean" -> Seq(
      CleanerHelper.removePunctuation,
      CleanerHelper.cleanDashes,
      CleanerHelper.cleanExtraWhitespace
    ),
    "html_clean" -> Seq(
      CleanerHelper.replaceUnicodeCharacters,
      CleanerHelper.cleanNonAsciiChars
    ),
    "full_auto" -> FUNCTIONAL_PRESETS.values.toSeq
  )

  // Reverse-map functions to preset names (for logging)
  private lazy val FUNCTION_NAME_LOOKUP: Map[String => String, String] = FUNCTIONAL_PRESETS.map(_.swap)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = annotations.map {
    annotation =>
      Try(
        applyDocumentNormalization(
          annotation.result,
          getAction,
          getPatterns,
          getReplacement,
          getPolicy,
          getLowercase,
          getEncoding)) match {
        case Success(cleanedDoc) =>
          Annotation(
            DOCUMENT,
            annotation.begin,
            cleanedDoc.length - 1,
            cleanedDoc,
            annotation.metadata)
        case Failure(_) =>
          Annotation.apply("")
      }
  }
}

/** This is the companion object of [[DocumentNormalizer]]. Please refer to that class for the
  * documentation.
  */
object DocumentNormalizer extends DefaultParamsReadable[DocumentNormalizer]

object LookAroundManager {

  val LOOKAHEAD_PATTERN = "(?="
  val LOOKBEHIND_PATTERN = "(?<="

  val SEMI_COLON = "\\;"
  val FULL_STOP = "\\.(?!\\d+)"
  val EXCLAMATION_MARK = "\\!"
  val QUESTION_MARK = "\\?"
  val END_FULL_STOPS_REGEX = "\\.$"
  val EMPTY_STR = ""
  val OR_STR = "|"

  def withReplacement(text: String, replacement: String, m: Regex.Match, groupIdx: Int = 1) = { // implicit condition of picking the
    // assuming first group to be the lookaround pattern replacement
    text.replace(m.group(groupIdx), replacement)
  }

  def process(text: String, patterns: Array[String], replacement: String): String = {
    // assuming first pattern to be a lookaround containing first group as replacement target
    val lookaheadPattern: String = patterns.head
    require(
      lookaheadPattern.contains(LOOKAHEAD_PATTERN) || lookaheadPattern.contains(
        LOOKBEHIND_PATTERN),
      "First pattern with action lookaround must contain a lookaround symbol, i.e. (?=criteria) or (?<=criteria)")

    val fullStopsTrimmed = text.replaceAll(END_FULL_STOPS_REGEX, EMPTY_STR)
    val separators = Array(SEMI_COLON, FULL_STOP, EXCLAMATION_MARK, QUESTION_MARK)

    val detectedSeps =
      for (s <- separators; if text.contains(s.replace("\\", ""))) yield s.replace("\\", "")

    val chunks =
      if (!detectedSeps.isEmpty)
        fullStopsTrimmed.split(detectedSeps.mkString(OR_STR))
      else
        Array(fullStopsTrimmed)

    val lookaheadRegex: Regex = lookaheadPattern.r

    val replacedChunks = new ListBuffer[String]()

    for (c <- chunks) {
      val res = lookaheadRegex.findFirstMatchIn(c) match {
        case Some(m) => withReplacement(c, replacement, m)
        case _ => c
      }
      replacedChunks += res
    }

    if (detectedSeps.length > 0)
      replacedChunks.mkString(detectedSeps.head)
    else
      replacedChunks.mkString
  }
}
