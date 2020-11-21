package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import org.apache.spark.ml.param.{BooleanParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}


/**
  * Annotator which normalizes raw text from tagged text, e.g. scraped web pages or xml documents, from document type columns into Sentence.
  * Removes all dirty characters from text following one or more input regex patterns.
  * Can apply non wanted character removal which a specific policy.
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

  /** normalization regex patterns which match will be removed from document
    *
    * @group Parameters
    **/
  val cleanupPatterns: StringArrayParam = new StringArrayParam(this, "cleanupPatterns", "normalization regex patterns which match will be removed from document. Defaults is \"<[^>]*>\"")

  /** whether to convert strings to lowercase
    *
    * @group param
    **/
  val lowercase = new BooleanParam(this, "lowercase", "whether to convert strings to lowercase")

  /** removalPolicy to remove patterns from text with a given policy
    *
    * @group param
    **/
  val removalPolicy: Param[String] = new Param(this, "removalPolicy", "removalPolicy to remove pattern from text")

  //  Assuming non-html does not contain any < or > and that input string is correctly structured
  setDefault(
    inputCols -> Array("document"),
    cleanupPatterns -> Array("<[^>]*>"),
    lowercase -> false,
    removalPolicy -> "pretty_all"
  )

  private val EMPTY_STR = ""
  private val BREAK_STR = "|##|"

  /** Regular expressions list for normalization
    * @group getParam
    **/
  def getCleanupPatterns: Array[String] = $(cleanupPatterns)

  /** Lowercase tokens, default true
    *
    * @group getParam
    **/
  def getLowercase: Boolean = $(lowercase)

  /** pattern to grab from text as token candidates. Defaults "pretty_all"
    *
    * @group getParam
    **/
  def getRemovalPolicy: String = $(removalPolicy)

  /** Regular expressions list for normalization,
    *
    *
    * @group setParam
    **/
  def setCleanupPatterns(value: Array[String]): this.type = set(cleanupPatterns, value)

  /** Lowercase tokens, default true
    *
    * @group setParam
    **/
  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  /** removal policy to apply
    * Valid policy values are: "all", "pretty_all", "first", "pretty_first"
    *
    * @group setParam
    **/
  def setRemovalPolicy(value: String): this.type = set(removalPolicy, value)

  private def withAllFormatter(text: String, replacement: String = EMPTY_STR): String ={
    val patternsStr: String = $(cleanupPatterns).mkString(BREAK_STR)
    text.replaceAll(patternsStr, replacement)
  }

  /** pattern to grab from text as token candidates. Defaults \\S+
    *
    **/
  private def withPrettyAllFormatter(text: String): String = {
    withAllFormatter(text).split("\\s+").map(_.trim).mkString(" ")
  }

  /** pattern to grab from text as token candidates. Defaults \\S+
    *
    **/
  private def withFirstFormatter(text: String, replacement: String = EMPTY_STR): String = {
    val patternsStr = $(cleanupPatterns).mkString(BREAK_STR)
    text.replaceFirst(patternsStr, replacement)
  }

  /** pattern to grab from text as token candidates. Defaults \\S+
    *
    **/
  private def withPrettyFirstFormatter(text: String): String = {
    withFirstFormatter(text).split("\\s+").map(_.trim).mkString(" ")
  }

  /** Apply patterns and removal policy
    *
    **/
  private def applyRegexPatterns(text: String, patterns: Array[String])(policy: String): String = {
    require(!text.isEmpty && patterns.length > 0 && !patterns(0).isEmpty && !policy.isEmpty)

    val cleaned: String = policy match {
      case "all" => withAllFormatter(text)
      case "pretty_all" => withPrettyAllFormatter(text)
      case "first" => withFirstFormatter(text)
      case "pretty_first" => withPrettyFirstFormatter(text)
      case _ => throw new Exception("Unknown policy parameter in DocumentNormalizer annotation." +
        "Please select either: all, pretty_all, first, or pretty_first")
    }

    if ($(lowercase)) cleaned.toLowerCase else cleaned
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.
      map { annotation =>
        val cleanedDoc = applyRegexPatterns(annotation.result, getCleanupPatterns)(getRemovalPolicy)
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
