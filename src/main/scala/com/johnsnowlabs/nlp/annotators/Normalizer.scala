package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/**
  * Annotator that cleans out tokens. Requires stems, hence tokens.
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
class Normalizer(override val uid: String) extends AnnotatorApproach[NormalizerModel] {

  /** Cleans out tokens */
  override val description: String = "Cleans out tokens"

  /** Output Annotator Type : TOKEN
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input Annotator Type : TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[String] = Array(TOKEN) // Annotator reference id. Used to identify elements in metadata or to refer to this annotator type


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
  /** delimited file with list of custom words to be manually corrected
    *
    * @group param
    **/
  val slangDictionary = new ExternalResourceParam(this, "slangDictionary", "delimited file with list of custom words to be manually corrected")
  /** whether or not to be case sensitive to match slangs. Defaults to false.
    *
    * @group param
    **/
  val slangMatchCase = new BooleanParam(this, "slangMatchCase", "whether or not to be case sensitive to match slangs. Defaults to false.")

  setDefault(
    lowercase -> false,
    cleanupPatterns -> Array("[^\\pL+]"),
    slangMatchCase -> false
  )

  /** Regular expressions list for normalization, defaults [^A-Za-z]
    *
    *
    * @group getParam
    **/
  def getCleanupPatterns: Array[String] = $(cleanupPatterns)

  /** Regular expressions list for normalization, defaults [^A-Za-z]
    *
    *
    * @group setParam
    **/
  def setCleanupPatterns(value: Array[String]): this.type = set(cleanupPatterns, value)

  /** Lowercase tokens, default true
    *
    * @group getParam
    **/
  def getLowercase: Boolean = $(lowercase)

  /** Lowercase tokens, default true
    *
    * @group setParam
    **/
  def setLowercase(value: Boolean): this.type = set(lowercase, value)

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

  /** Txt file with delimited words to be transformed into something else
    *
    * @group setParam
    **/
  def setSlangDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "slang dictionary is a delimited text. needs 'delimiter' in options")
    set(slangDictionary, value)
  }

  /** Txt file with delimited words to be transformed into something else
    *
    * @group setParam
    **/
  def setSlangDictionary(path: String,
                         delimiter: String,
                         readAs: ReadAs.Format = ReadAs.TEXT,
                         options: Map[String, String] = Map("format" -> "text")): this.type =
    set(slangDictionary, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NormalizerModel = {

    val loadSlangs = if (get(slangDictionary).isDefined) {
      val parsed = ResourceHelper.parseKeyValueText($(slangDictionary))
      if ($(slangMatchCase))
        parsed.mapValues(_.trim)
      else
        parsed.map{case (k, v) => (k.toLowerCase, v.trim)}
    }
    else
      Map.empty[String, String]

    new NormalizerModel()
      .setCleanupPatterns($(cleanupPatterns))
      .setLowercase($(lowercase))
      .setSlangDict(loadSlangs)
      .setSlangMatchCase($(slangMatchCase))
  }

}

object Normalizer extends DefaultParamsReadable[Normalizer]