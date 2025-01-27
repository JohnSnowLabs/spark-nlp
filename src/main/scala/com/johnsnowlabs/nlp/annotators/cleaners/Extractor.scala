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

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

import scala.util.matching.Regex

class Extractor(override val uid: String)
    extends AnnotatorModel[Extractor]
    with HasSimpleAnnotate[Extractor] {

  def this() = this(Identifiable.randomUID("Extractor"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = CHUNK

  private val EMAIL_DATETIMETZ_PATTERN =
    "[A-Za-z]{3},\\s\\d{1,2}\\s[A-Za-z]{3}\\s\\d{4}\\s\\d{2}:\\d{2}:\\d{2}\\s[+-]\\d{4}"
  private val EMAIL_ADDRESS_PATTERN = "[a-z0-9\\.\\-+_]+@[a-z0-9\\.\\-+_]+\\.[a-z]+"

  private val IPV4_PATTERN: String =
    """(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}"""
  private val IPV6_PATTERN: String =
    """[a-z0-9]{4}::[a-z0-9]{4}:[a-z0-9]{4}:[a-z0-9]{4}:[a-z0-9]{4}%?[0-9]*"""
  private val IP_ADDRESS_PATTERN: String = s"($IPV4_PATTERN|$IPV6_PATTERN)"
  private val IP_ADDRESS_NAME_PATTERN = "[a-zA-Z0-9-]*\\.[a-zA-Z]*\\.[a-zA-Z]*"

  private val MAPI_ID_PATTERN = "[0-9]*\\.[0-9]*\\.[0-9]*\\.[0-9]*;"
  private val US_PHONE_NUMBERS_PATTERN =
    "(?:\\+?(\\d{1,3}))?[-. (]*(\\d{3})?[-. )]*(\\d{3})[-. ]*(\\d{4})(?: *x(\\d+))?\\s*$"

  private val IMAGE_URL_PATTERN =
    """(?i)https?://(?:[a-z0-9$_@.&+!*\\(\\),%-])+(?:/[a-z0-9$_@.&+!*\\(\\),%-]*)*\.(?:jpg|jpeg|png|gif|bmp|heic)"""

  val emailDateTimeTzPattern = new Param[String](
    this,
    "emailDateTimeTzPattern",
    "Specifies the date-time pattern for email timestamps, including time zone formatting.")

  /** @group setParam */
  def setEmailDateTimeTzPattern(value: String): this.type = set(emailDateTimeTzPattern, value)

  val emailAddress =
    new Param[String](this, "emailAddress", "Specifies the pattern for email addresses.")

  val ipAddressPattern =
    new Param[String](this, "ipAddressPattern", "Specifies the pattern for IP addresses.")

  /** @group setParam */
  def setIPAddressPattern(value: String): this.type = set(ipAddressPattern, value)

  val ipAddressNamePattern = new Param[String](
    this,
    "ipAddressNamePattern",
    "Specifies the pattern for IP addresses with names.")

  /** @group setParam */
  def setIpAddressNamePattern(value: String): this.type = set(ipAddressNamePattern, value)

  val mapiIdPattern =
    new Param[String](this, "mapiIdPattern", "Specifies the pattern for MAPI IDs.")

  /** @group setParam */
  def setMapiIdPattern(value: String): this.type = set(mapiIdPattern, value)

  val usPhoneNumbersPattern = new Param[String](
    this,
    "usPhoneNumbersPattern",
    "Specifies the pattern for US phone numbers.")

  val imageUrlPattern =
    new Param[String](this, "imageUrlPattern", "Specifies the pattern for image URLs.")

  /** @group setParam */
  def setImageUrlPattern(value: String): this.type = set(imageUrlPattern, value)

  val textPattern =
    new Param[String](this, "textPattern", "Specifies the pattern for text after and before.")

  def setTextPattern(value: String): this.type = set(textPattern, value)

  /** extractor can take the following values:
    *   - `email_date`: extract email date
    *   - `email_address`: extract email address
    *   - `ip_address`: extract ip address
    *   - `ip_address_name`: extract ip address with name
    *   - `mapi_id`: extract mapi id
    *   - `us_phone_numbers`: extract US phone numbers
    *   - `image_urls`: extract image URLs
    *   - `bullets`: extract ordered bullets
    * @group param
    */
  val extractorMode: Param[String] = new Param[String](
    this,
    "extractorMode",
    "possible values: " +
      "email_date, email_address, ip_address, ip_address_name, mapi_id, us_phone_numbers, image_urls, bullets, text_after, text_before")

  /** @group setParam */
  def setExtractorMode(value: String): this.type = {
    value.trim.toLowerCase() match {
      case "email_date" => set(extractorMode, "email_date")
      case "email_address" => set(extractorMode, "email_address")
      case "ip_address" => set(extractorMode, "ip_address")
      case "ip_address_name" => set(extractorMode, "ip_address_name")
      case "mapi_id" => set(extractorMode, "mapi_id")
      case "us_phone_numbers" => set(extractorMode, "us_phone_numbers")
      case "image_urls" => set(extractorMode, "image_urls")
      case "bullets" => set(extractorMode, "bullets")
      case "text_after" => set(extractorMode, "text_after")
      case "text_before" => set(extractorMode, "text_before")
      case _ => throw new IllegalArgumentException(s"Extractor mode $value not supported.")
    }
    set(extractorMode, value)
  }

  setDefault(
    emailDateTimeTzPattern -> EMAIL_DATETIMETZ_PATTERN,
    emailAddress -> EMAIL_ADDRESS_PATTERN,
    ipAddressPattern -> IP_ADDRESS_PATTERN,
    ipAddressNamePattern -> IP_ADDRESS_NAME_PATTERN,
    mapiIdPattern -> MAPI_ID_PATTERN,
    usPhoneNumbersPattern -> US_PHONE_NUMBERS_PATTERN,
    imageUrlPattern -> IMAGE_URL_PATTERN,
    extractorMode -> "undefined")

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    require($(extractorMode) != "undefined", "Extractor mode must be set.")

    $(extractorMode) match {
      case "email_date" => extractRegexPattern(annotations, $(emailDateTimeTzPattern).r)
      case "email_address" => extractRegexPattern(annotations, $(emailAddress).r)
      case "ip_address" => extractRegexPattern(annotations, $(ipAddressPattern).r)
      case "ip_address_name" => extractRegexPattern(annotations, $(ipAddressNamePattern).r)
      case "mapi_id" => extractRegexPattern(annotations, $(mapiIdPattern).r)
      case "us_phone_numbers" => extractRegexPattern(annotations, $(usPhoneNumbersPattern).r)
      case "image_urls" => extractImageUrls(annotations, $(imageUrlPattern).r)
      case "bullets" =>
        annotations.map { annotation =>
          extractOrderedBulletsAsAnnotation(annotation.result)
        }
      case "text_after" =>
        annotations.map { annotation =>
          extractTextAfter(annotation.result, $(textPattern))
        }
      case "text_before" =>
        annotations.map { annotation =>
          extractTextBefore(annotation.result, $(textPattern))
        }
      case _ =>
        throw new IllegalArgumentException(s"Extractor mode ${$(extractorMode)} not supported.")
    }

  }

  private def extractImageUrls(annotations: Seq[Annotation], regex: Regex): Seq[Annotation] = {
    annotations.flatMap { annotation =>
      regex.findAllMatchIn(annotation.result).map { matched =>
        val start = annotation.begin + matched.start
        val end = annotation.begin + matched.end - 1
        Annotation(outputAnnotatorType, start, end, matched.matched, Map.empty)
      }
    }
  }

  private def extractRegexPattern(annotations: Seq[Annotation], regex: Regex): Seq[Annotation] = {
    annotations.flatMap { annotation =>
      regex.findAllMatchIn(annotation.result).map { matched =>
        val start = annotation.begin + matched.start
        val end = annotation.begin + matched.end - 1
        Annotation(outputAnnotatorType, start, end, matched.matched, Map.empty)
      }
    }
  }

  /** Extracts the start of bulleted text sections, considering numeric and alphanumeric types,
    * and returns the result as an Annotation.
    *
    * @param text
    *   The input string.
    * @return
    *   An Annotation object containing extracted bullet information.
    *
    * Example:
    * ------- "This is a very important point" -> Annotation("bullet", 0, 0, "None,None,None",
    * Map.empty) "1.1 This is a very important point" -> Annotation("bullet", 0, 3, "1,1,None",
    * Map("section" -> "1", "sub_section" -> "1")) "a.1 This is a very important point" ->
    * Annotation("bullet", 0, 3, "a,1,None", Map("section" -> "a", "sub_section" -> "1"))
    */
  private def extractOrderedBulletsAsAnnotation(text: String): Annotation = {
    var section: Option[String] = None
    var subSection: Option[String] = None
    var subSubSection: Option[String] = None

    val textParts = text.split("\\s+", 2)

    val defaultBegin = 0
    val defaultEnd = 0

    if (textParts.isEmpty || textParts.head.count(_ == '.') == 0 || textParts.head.contains(
        "..")) {
      return Annotation(
        annotatorType = outputAnnotatorType,
        begin = defaultBegin,
        end = defaultEnd,
        result = "None,None,None",
        metadata = Map.empty)
    }

    val bulletPattern: Regex = "\\.".r
    val bulletParts = bulletPattern.split(textParts.head).filter(_.nonEmpty)

    if (bulletParts.headOption.exists(_.length > 2)) {
      return Annotation(
        annotatorType = outputAnnotatorType,
        begin = defaultBegin,
        end = defaultEnd,
        result = "None,None,None",
        metadata = Map.empty)
    }

    val begin = 0
    val end = textParts.head.length

    section = Some(bulletParts.head)
    if (bulletParts.length > 1) {
      subSection = Some(bulletParts(1))
    }
    if (bulletParts.length > 2) {
      subSubSection = Some(bulletParts(2))
    }

    val result =
      s"(${section.getOrElse("None")},${subSection.getOrElse("None")},${subSubSection.getOrElse("None")})"
    val metadata = Map(
      "section" -> section.getOrElse("None"),
      "sub_section" -> subSection.getOrElse("None"),
      "sub_sub_section" -> subSubSection.getOrElse("None")).filterNot(_._2 == "None")

    Annotation(
      annotatorType = outputAnnotatorType,
      begin = begin,
      end = end,
      result = result,
      metadata = metadata)
  }

  /** Extracts text that occurs after the specified pattern and returns an Annotation.
    *
    * @param text
    *   The input text.
    * @param pattern
    *   The regex pattern to search for.
    * @param index
    *   The occurrence index of the pattern.
    * @param strip
    *   If true, removes leading whitespace from the extracted string.
    * @return
    *   Annotation with details of the extracted result.
    */
  private def extractTextAfter(
      text: String,
      pattern: String,
      index: Int = 0,
      strip: Boolean = true): Annotation = {
    val regexMatch = getIndexedMatch(text, pattern, index)
    val begin = regexMatch.end
    val afterText = text.substring(begin)
    val result = if (strip) afterText.replaceAll("^\\s+", "") else afterText

    Annotation(
      annotatorType = outputAnnotatorType,
      begin = begin,
      end = text.length,
      result = result,
      metadata = Map("index" -> index.toString))
  }

  /** Extracts text that occurs before the specified pattern and returns an Annotation.
    *
    * @param text
    *   The input text.
    * @param pattern
    *   The regex pattern to search for.
    * @param index
    *   The occurrence index of the pattern.
    * @param strip
    *   If true, removes trailing whitespace from the extracted string.
    * @return
    *   Annotation with details of the extracted result.
    */
  private def extractTextBefore(
      text: String,
      pattern: String,
      index: Int = 0,
      strip: Boolean = true): Annotation = {
    val regexMatch = getIndexedMatch(text, pattern, index)
    val start = regexMatch.start
    val beforeText = text.substring(0, start)
    val result = if (strip) beforeText.replaceAll("\\s+$", "") else beforeText

    Annotation(
      annotatorType = outputAnnotatorType,
      begin = 0,
      end = start,
      result = result,
      metadata = Map("index" -> index.toString))
  }

  private def getIndexedMatch(text: String, pattern: String, index: Int = 0): Regex.Match = {
    if (index < 0)
      throw new IllegalArgumentException(
        s"The index is $index. Index must be a non-negative integer.")

    val regex = new Regex(pattern)
    val matches = regex.findAllMatchIn(text).toSeq

    if (index >= matches.length)
      throw new IllegalArgumentException(
        s"Result with index $index was not found. The largest index was ${matches.length - 1}.")

    matches(index)
  }

}
