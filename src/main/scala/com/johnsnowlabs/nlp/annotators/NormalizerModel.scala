package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable, WithAnnotate}
import org.apache.spark.ml.param.{BooleanParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable

/**
  * Annotator that cleans out tokens. Requires stems, hence tokens.
  *
  * Removes all dirty characters from text following a regex pattern and transforms words based on a provided dictionary
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/NormalizerTestSpec.scala]] for examples on how to use the API
  *
  * @param uid required internal uid for saving annotator
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
class NormalizerModel(override val uid: String) extends AnnotatorModel[NormalizerModel] with WithAnnotate[NormalizerModel] {

  /** Output annotator type : TOKEN
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = TOKEN
  /** Input annotator type : TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  case class TokenizerAndNormalizerMap(beginTokenizer: Int, endTokenizer: Int, token: String,
                                       beginNormalizer: Int, endNormalizer: Int, normalizer: String)

  /** normalization regex patterns which match will be removed from token
    *
    * @group param
    **/
  val cleanupPatterns = new StringArrayParam(this, "cleanupPatterns", "normalization regex patterns which match will be removed from token")
  /** whether to convert strings to lowercase
    *
    * @group param
    **/
  val lowercase = new BooleanParam(this, "lowercase", "whether to convert strings to lowercase")
  /** slangDict
    *
    * @group param
    **/
  protected val slangDict: MapFeature[String, String] = new MapFeature(this, "slangDict")
  /** whether or not to be case sensitive to match slangs. Defaults to false.
    *
    * @group param
    **/
  val slangMatchCase = new BooleanParam(this, "slangMatchCase", "whether or not to be case sensitive to match slangs. Defaults to false.")

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  /** Regular expressions list for normalization, defaults [^A-Za-z]
    * @group setParam
    **/
  def setCleanupPatterns(value: Array[String]): this.type = set(cleanupPatterns, value)

  /** Regular expressions list for normalization, defaults [^A-Za-z]
    * @group setParam
    **/
  def getCleanupPatterns: Array[String] = $(cleanupPatterns)

  /** Lowercase tokens, default true
    *
    * @group setParam
    **/
  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  /** Lowercase tokens, default true
    *
    * @group setParam
    **/
  def getLowercase: Boolean = $(lowercase)


  /** Txt file with delimited words to be transformed into something else
    *
    * @group setParam
    **/
  def setSlangDict(value: Map[String, String]): this.type = set(slangDict, value)

  /** Whether to convert string to lowercase or not while checking
    *
    * @group setParam
    **/
  def setSlangMatchCase(value: Boolean): this.type = set(slangMatchCase, value)

  /** Whether to convert string to lowercase or not while checking
    *
    * @group getParam
    **/
  def getSlangMatchCase: Boolean = $(slangMatchCase)

  def applyRegexPatterns(word: String): String = {

    val nToken = {
      get(cleanupPatterns).map(_.foldLeft(word)((currentText, compositeToken) => {
        currentText.replaceAll(compositeToken, "")
      })).getOrElse(word)
    }
    nToken
  }

  /** Txt file with delimited words to be transformed into something else
    *
    * @group getParam
    **/
  protected def getSlangDict: Map[String, String] = $$(slangDict)

  /** ToDo: Review implementation, Current implementation generates spaces between non-words, potentially breaking tokens */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val normalizedAnnotations = annotations.flatMap { originalToken =>

      /** slang dictionary keys should have been lowercased if slangMatchCase is false */
      val unslanged = $$(slangDict).get(
        if ($(slangMatchCase)) originalToken.result
        else originalToken.result.toLowerCase
      )

      /** simple-tokenize the unslanged slag phrase */
      val tokenizedUnslang = {
        unslanged.map(unslang => {
          unslang.split(" ")
        }).getOrElse(Array(originalToken.result))
      }

      val cleaned = tokenizedUnslang.map(word => applyRegexPatterns(word))

      val cased = if ($(lowercase)) cleaned.map(_.toLowerCase) else cleaned

      cased.filter(_.nonEmpty).map { finalToken => {
        Annotation(
          outputAnnotatorType,
          originalToken.begin,
          originalToken.begin + finalToken.length - 1,
          finalToken,
          originalToken.metadata
        )
      }}

    }

    normalizedAnnotations

  }

}

object NormalizerModel extends ParamsAndFeaturesReadable[NormalizerModel]