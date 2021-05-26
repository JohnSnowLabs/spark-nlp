package com.johnsnowlabs.nlp.annotators.sda.pragmatic

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, SENTIMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, DoubleParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/**
  * Gives a good or bad score to a sentence based on the approach used
  *
  * @param uid internal uid needed for saving annotator to disk
  * @@ model: Implementation to be applied for sentiment
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
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class SentimentDetector(override val uid: String) extends AnnotatorApproach[SentimentDetectorModel] {

  /** Output annotation type : SENTIMENT
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = SENTIMENT
  /** Input annotation type : TOKEN, DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /** Rule based sentiment detector */
  override val description: String = "Rule based sentiment detector"

  def this() = this(Identifiable.randomUID("SENTIMENT"))

  /** multiplier for positive sentiments. Defaults 1.0
    *
    * @group param
    **/
  val positiveMultiplier = new DoubleParam(this, "positiveMultiplier", "multiplier for positive sentiments. Defaults 1.0")
  /** "multiplier for negative sentiments. Defaults -1.0
    *
    * @group param
    **/
  val negativeMultiplier = new DoubleParam(this, "negativeMultiplier", "multiplier for negative sentiments. Defaults -1.0")
  /** multiplier for increment sentiments. Defaults 2.0
    *
    * @group param
    **/
  val incrementMultiplier = new DoubleParam(this, "incrementMultiplier", "multiplier for increment sentiments. Defaults 2.0")
  /** multiplier for decrement sentiments. Defaults -2.0
    *
    * @group param
    **/
  val decrementMultiplier = new DoubleParam(this, "decrementMultiplier", "multiplier for decrement sentiments. Defaults -2.0")
  /** multiplier for revert sentiments. Defaults -1.0
    *
    * @group param
    **/
  val reverseMultiplier = new DoubleParam(this, "reverseMultiplier", "multiplier for revert sentiments. Defaults -1.0")
  /** if true, score will show as the double value, else will output string \"positive\" or \"negative\". Defaults false
    *
    * @group param
    **/
  val enableScore = new BooleanParam(this, "enableScore", "if true, score will show as the double value, else will output string \"positive\" or \"negative\". Defaults false")
  /** delimited file with a list sentiment tags per word. Requires 'delimiter' in options
    *
    * @group param
    **/
  val dictionary = new ExternalResourceParam(this, "dictionary", "delimited file with a list sentiment tags per word. Requires 'delimiter' in options")

  setDefault(
    positiveMultiplier -> 1.0,
    negativeMultiplier -> -1.0,
    incrementMultiplier -> 2.0,
    decrementMultiplier -> -2.0,
    reverseMultiplier -> -1.0,
    enableScore -> false
  )

  /** Multiplier for positive sentiments. Defaults 1.0
    *
    * @group param
    **/
  def setPositiveMultiplier(v: Double): this.type = set(positiveMultiplier, v)

  /** Multiplier for negative sentiments. Defaults -1.0
    *
    * @group param
    **/
  def setNegativeMultiplier(v: Double): this.type = set(negativeMultiplier, v)

  /** Multiplier for increment sentiments. Defaults 2.0
    *
    * @group param
    **/
  def setIncrementMultiplier(v: Double): this.type = set(incrementMultiplier, v)

  /** Multiplier for decrement sentiments. Defaults -2.0
    *
    * @group param
    **/
  def setDecrementMultiplier(v: Double): this.type = set(decrementMultiplier, v)

  /** Multiplier for revert sentiments. Defaults -1.0
    *
    * @group param
    **/
  def setReverseMultiplier(v: Double): this.type = set(reverseMultiplier, v)

  /** if true, score will show as the double value, else will output string \"positive\" or \"negative\". Defaults false
    *
    * @group param
    **/
  def setEnableScore(v: Boolean): this.type = set(enableScore, v)

  /** delimited file with a list sentiment tags per word. Requires 'delimiter' in options. Dictionary needs 'delimiter' in order to separate words from sentiment tags
    *
    * @group param
    **/
  def setDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "dictionary needs 'delimiter' in order to separate words from sentiment tags")
    set(dictionary, value)
  }

  /** delimited file with a list sentiment tags per word. Requires 'delimiter' in options. Dictionary needs 'delimiter' in order to separate words from sentiment tags
    *
    * @group param
    **/
  def setDictionary(path: String,
                    delimiter: String,
                    readAs: ReadAs.Format,
                    options: Map[String, String] = Map("format" -> "text")): this.type =
    set(dictionary, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SentimentDetectorModel = {
    new SentimentDetectorModel()
      .setIncrementMultipler($(incrementMultiplier))
      .setDecrementMultipler($(decrementMultiplier))
      .setPositiveMultipler($(positiveMultiplier))
      .setNegativeMultipler($(negativeMultiplier))
      .setReverseMultipler($(reverseMultiplier))
      .setEnableScore($(enableScore))
      .setSentimentDict(ResourceHelper.parseKeyValueText($(dictionary)))
  }

}
object SentimentDetector extends DefaultParamsReadable[SentimentDetector]
