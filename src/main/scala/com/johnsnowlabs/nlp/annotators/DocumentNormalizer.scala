package com.johnsnowlabs.nlp.annotators

import java.nio.charset.{Charset, StandardCharsets}

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import org.apache.spark.ml.param.{BooleanParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.xml.XML


/**
  * Annotator which normalizes raw text from tagged text, e.g. scraped web pages or xml documents, from document type columns into Sentence.
  * Removes all dirty characters from text following one or more input regex patterns.
  * Can apply not wanted character removal with a specific policy.
  * Can apply lower case normalization.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DocumentNormalizerTestSpec.scala DocumentNormalizer test class]] for examples examples of usage.
  *
  * @param uid required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
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
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class DocumentNormalizer(override val uid: String) extends AnnotatorModel[DocumentNormalizer] {

  private val EMPTY_STR = ""
  private val BREAK_STR = "|##|"
  private val SPACE_STR = " "
  private val GENERIC_TAGS_REMOVAL_PATTERN = "<[^>]*>"

  /** Input annotator type : DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  /** Input annotator type : DOCUMENT
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  def this() = this(Identifiable.randomUID("DOCUMENT_NORMALIZER"))

  /** action to perform applying regex patterns on text
    *
    * @group param
    **/
  val action: Param[String] = new Param(this, "action", "action to perform applying regex patterns on text")

  /** normalization regex patterns which match will be removed from document
    *
    * @group Parameters
    **/
  val patterns: StringArrayParam = new StringArrayParam(this, "patterns", "normalization regex patterns which match will be removed from document. Defaults is \"<[^>]*>\"")

  /** replacement string to apply when regexes match
    *
    * @group param
    **/
  val replacement: Param[String] = new Param(this, "replacement", "replacement string to apply when regexes match")

  /** whether to convert strings to lowercase
    *
    * @group param
    **/
  val lowercase = new BooleanParam(this, "lowercase", "whether to convert strings to lowercase")

  /** removalPolicy to remove patterns from text with a given policy
    *
    * @group param
    **/
  val policy: Param[String] = new Param(this, "policy", "removalPolicy to remove pattern from text")

  /** file encoding to apply on normalized documents
    *
    * @group param
    **/
  val encoding: Param[String] = new Param(this, name = "encoding", "file encoding to apply on normalized documents")

  //  Assuming non-html does not contain any < or > and that input string is correctly structured
  setDefault(
    inputCols -> Array(AnnotatorType.DOCUMENT),
    action -> "clean",
    patterns -> Array(GENERIC_TAGS_REMOVAL_PATTERN),
    replacement -> SPACE_STR,
    lowercase -> false,
    policy -> "pretty_all",
    encoding -> "disable"
  )

  /** Action to perform on text. Default "clean".
    *
    * @group getParam
    **/
  def getAction: String = $(action)

  /** Regular expressions list for normalization.
    *
    * @group getParam
    **/
  def getPatterns: Array[String] = $(patterns)

  /** Replacement string to apply when regexes match
    *
    * @group getParam
    **/
  def getReplacement: String = $(replacement)

  /** Lowercase tokens, default false
    *
    * @group getParam
    **/
  def getLowercase: Boolean = $(lowercase)

  /** Policy to remove patterns from text. Defaults "pretty_all".
    *
    * @group getParam
    **/
  def getPolicy: String = $(policy)

  /** Encoding to apply to normalized documents.
    *
    * @group getParam
    **/
  def getEncoding: String = $(encoding)

  /** Action to perform on text. Default "clean".
    *
    * @group getParam
    **/
  def setAction(value: String): this.type = set(action, value)

  /** Regular expressions list for normalization.
    *
    * @group setParam
    **/
  def setPatterns(value: Array[String]): this.type = set(patterns, value)

  /** Replacement string to apply when regexes match
    *
    * @group getParam
    **/
  def setReplacement(value: String): this.type = set(replacement, value)

  /** Lower case tokens, default false
    *
    * @group setParam
    **/
  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  /** Removal policy to apply.
    * Valid policy values are: "all", "pretty_all", "first", "pretty_first"
    *
    * @group setParam
    **/
  def setPolicy(value: String): this.type = set(policy, value)

  /** Encoding to apply. Default is UTF-8.
    * Valid encoding are values are: UTF_8, UTF_16, US_ASCII, ISO-8859-1, UTF-16BE, UTF-16LE
    *
    * @group setParam
    **/
  def setEncoding(value: String): this.type = set(encoding, value)

  /** Applying document normalization without pretty formatting (removing multiple spaces)
    *
    **/
  private def withAllFormatter(text: String,
                               action: String,
                               patterns: Array[String],
                               replacement: String): String = {
    action match {
      case "clean" =>
        val patternsStr: String = patterns.mkString(BREAK_STR)
        text.replaceAll(patternsStr, replacement)
      case "extract" => {
        val htmlXml = XML.loadString(text)
        val textareaContents = (htmlXml \\ patterns.mkString).text
        textareaContents
      }
      case _ => throw new Exception("Unknown action parameter in DocumentNormalizer annotation." +
        "Please select either: clean or extract")
    }
  }

  /** pattern to grab from text as token candidates. Defaults \\S+
    *
    **/
  private def withPrettyAllFormatter(text: String,
                                     action: String,
                                     patterns: Array[String],
                                     replacement: String): String = {
    withAllFormatter(text, action, patterns, replacement)
      .split("\\s+").map(_.trim).mkString(SPACE_STR)
  }

  /** Applying document normalization without pretty formatting (removing multiple spaces) retrieving first element only
    *
    **/
  private def withFirstFormatter(text: String,
                                 action: String,
                                 patterns: Array[String],
                                 replacement: String): String = {
    action match {
      case "clean" =>
        val patternsStr: String = patterns.mkString(BREAK_STR)
        text.replaceFirst(patternsStr, replacement)
      case "extract" => {
        val htmlXml = XML.loadString(text)
        val textareaContents = htmlXml \\ patterns.mkString
        textareaContents.head.mkString
      }
      case _ => throw new Exception("Unknown action parameter in DocumentNormalizer annotation." +
        "Please select either: clean or extract")
    }
  }

  /** pattern to grab from text as token candidates. Defaults \\S+
    *
    **/
  private def withPrettyFirstFormatter(text: String,
                                       action: String,
                                       patterns: Array[String],
                                       replacement: String): String = {
    withFirstFormatter(text, action, patterns, replacement)
      .split("\\s+").map(_.trim).mkString(SPACE_STR)
  }

  /** Apply a given encoding to the processed text
    *
    * US-ASCII
    * Seven-bit ASCII, a.k.a. ISO646-US, a.k.a. the Basic Latin block of the Unicode character set
    *
    * ISO-8859-1
    * ISO Latin Alphabet No. 1, a.k.a. ISO-LATIN-1
    *
    * UTF-8
    * Eight-bit UCS Transformation Format
    *
    * UTF-16BE
    * Sixteen-bit UCS Transformation Format, big-endian byte order
    *
    * UTF-16LE
    * Sixteen-bit UCS Transformation Format, little-endian byte order
    *
    * UTF-16
    * Sixteen-bit UCS Transformation Format, byte order identified by an optional byte-order mark
    **/
  private def withEncoding(text: String, encoding: Charset = StandardCharsets.UTF_8): String ={
    val defaultCharset: Charset = Charset.defaultCharset
    if(!Charset.defaultCharset.equals(encoding)){
      log.warn("Requested encoding parameter is different from the default charset.")
    }
    new String(text.getBytes(defaultCharset), encoding)
  }

  /** Apply document normalization on text using action, patterns, policy, lowercase and encoding parameters.
    *
    */
  private def applyDocumentNormalization(text: String,
                                         action: String,
                                         patterns: Array[String],
                                         replacement: String,
                                         policy: String,
                                         lowercase: Boolean,
                                         encoding: String): String = {
    require(!text.isEmpty && !action.isEmpty && patterns.length > 0 && !patterns(0).isEmpty && !policy.isEmpty)

    val processedWithActionPatterns: String = policy match {
      case "all" => withAllFormatter(text, action, patterns, replacement)
      case "pretty_all" => withPrettyAllFormatter(text, action, patterns, replacement)
      case "first" => withFirstFormatter(text, action, patterns, replacement)
      case "pretty_first" => withPrettyFirstFormatter(text, action, patterns, replacement)
      case _ => throw new Exception("Unknown policy parameter in DocumentNormalizer annotation." +
        "Please select either: all, pretty_all, first, or pretty_first")
    }

    val cased =
      if (lowercase)
        processedWithActionPatterns.toLowerCase
      else
        processedWithActionPatterns

    encoding match {
      case "disable" => cased
      case "UTF-8" => withEncoding(cased, StandardCharsets.UTF_8)
      case "UTF-16" => withEncoding(cased, StandardCharsets.UTF_16)
      case "US-ASCII" => withEncoding(cased, StandardCharsets.US_ASCII)
      case "ISO-8859-1" => withEncoding(cased, StandardCharsets.ISO_8859_1)
      case "UTF-16BE" => withEncoding(cased, StandardCharsets.UTF_16BE)
      case "UTF-16LE" => withEncoding(cased, StandardCharsets.UTF_16LE)
      case _ => throw new Exception("Unknown encoding parameter in DocumentNormalizer annotation." +
        "Please select either: disable, UTF_8, UTF_16, US_ASCII, ISO-8859-1, UTF-16BE, UTF-16LE")
    }
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.
      map { annotation =>
        val cleanedDoc =
          applyDocumentNormalization(
            annotation.result,
            getAction,
            getPatterns,
            getReplacement,
            getPolicy,
            getLowercase,
            getEncoding)

        Annotation(
          DOCUMENT,
          annotation.begin,
          cleanedDoc.length - 1,
          cleanedDoc,
          annotation.metadata
        )
      }
  }
}

object DocumentNormalizer extends DefaultParamsReadable[DocumentNormalizer]
