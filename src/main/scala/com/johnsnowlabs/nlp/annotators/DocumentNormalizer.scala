package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset


/**
  * Annotator which normalizes raw text from tagged text, e.g. scraped web pages or xml documents, from document type columns into Sentence.
  * Removes all dirty characters from text following one or more input regex patterns.
  * Can apply non wanted character removal which a specific policy.
  * Can apply lower case normalization.
  *
  * Identifies tokens with tokenization open standards. A few rules will help customizing it if defaults do not fit user needs.
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
class DocumentNormalizer(override val uid: String) extends AnnotatorApproach[DocumentNormalizerModel] {

  override val description: String = "Annotator that cleans out text tag contents from documents"

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
  val cleanupPatterns: StringArrayParam = new StringArrayParam(this, "cleanupPatterns", "normalization regex patterns which match will be removed from document")

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
    cleanupPatterns -> Array("<[^>]*>"),
    lowercase -> true,
    removalPolicy -> "pretty_all"
  )

  /** Regular expressions list for normalization, defaults "<[^>]*>".
    *
    *
    * @group getParam
    **/
  def getCleanupPatterns: Array[String] = $(cleanupPatterns)

  /** Regular expressions list for normalization, defaults "<[^>]*>".
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

  /** Lowercase tokens, default true
    *
    * @group getParam
    **/
  def getLowercase: Boolean = $(lowercase)

  /** removal policy to apply
    * Valid policy values are: "all", "pretty_all", "first", "pretty_first"
    *
    * @group setParam
    **/
  def setRemovalPolicy(value: String): this.type = set(removalPolicy, value)

  /** pattern to grab from text as token candidates. Defaults "all"
    *
    * @group getParam
    **/
  def getRemovalPolicy: String = $(removalPolicy)

  /**
    *  Clears out rules and constructs a new rule for every combination of rules provided.
    *  The strategy is to catch one token per regex group.
    *  User may add its own groups if needs targets to be tokenized separately from the rest.
    *
    *  @param dataset
    *  @param recursivePipeline
    *  @return TokenizedModel
    *
    */
  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DocumentNormalizerModel = {
    new DocumentNormalizerModel()
      .setCleanUpPatterns($(cleanupPatterns))
      .setLowercase($(lowercase))
      .setRemovalPolicy($(removalPolicy))
  }
}


object DocumentNormalizer extends DefaultParamsReadable[DocumentNormalizer]