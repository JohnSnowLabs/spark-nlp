package com.johnsnowlabs.nlp.annotators.sda.pragmatic

import com.johnsnowlabs.nlp.annotators.common.TokenizedWithSentence
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable, HasSimpleAnnotate}
import org.apache.spark.ml.param.{BooleanParam, DoubleParam}
import org.apache.spark.ml.util.Identifiable

/**
  * Created by saif on 12/06/2017.
  */

/**
  * Gives a good or bad score to a sentence based on the approach used
  *
  * @param uid internal uid needed for saving annotator to disk
  * @@ model: Implementation to be applied for sentiment analysis
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
class SentimentDetectorModel(override val uid: String) extends AnnotatorModel[SentimentDetectorModel] with HasSimpleAnnotate[SentimentDetectorModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Sentiment dict
    *
    * @group param
    **/
  val sentimentDict = new MapFeature[String, String](this, "sentimentDict")

  /** @group param */
  lazy val model: PragmaticScorer = new PragmaticScorer($$(sentimentDict), $(positiveMultiplier), $(negativeMultiplier), $(incrementMultiplier), $(decrementMultiplier), $(reverseMultiplier))

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

  def this() = this(Identifiable.randomUID("SENTIMENT"))

  /** multiplier for positive sentiments. Defaults 1.0
    *
    * @group param
    **/
  val positiveMultiplier = new DoubleParam(this, "positiveMultiplier", "multiplier for positive sentiments. Defaults 1.0")
  /** multiplier for negative sentiments. Defaults -1.0
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
  /** if true, score will show as a string type containing a double value, else will output string \"positive\" or \"negative\". Defaults false
    *
    * @group param
    **/
  val enableScore = new BooleanParam(this, "enableScore", "if true, score will show as a string type containing a double value, else will output string \"positive\" or \"negative\". Defaults false")

  /** Multiplier for positive sentiments. Defaults 1.0
    *
    * @group setParam
    **/
  def setPositiveMultipler(v: Double): this.type = set(positiveMultiplier, v)

  /** Multiplier for negative sentiments. Defaults -1.0
    *
    * @group setParam
    **/
  def setNegativeMultipler(v: Double): this.type = set(negativeMultiplier, v)

  /** Multiplier for increment sentiments. Defaults 2.0
    *
    * @group setParam
    **/
  def setIncrementMultipler(v: Double): this.type = set(incrementMultiplier, v)

  /** Multiplier for decrement sentiments. Defaults -2.0
    *
    * @group setParam
    **/
  def setDecrementMultipler(v: Double): this.type = set(decrementMultiplier, v)

  /** Multiplier for revert sentiments. Defaults -1.0
    *
    * @group setParam
    **/
  def setReverseMultipler(v: Double): this.type = set(reverseMultiplier, v)

  /** If true, score will show as a string type containing a double value, else will output string "positive" or "negative". Defaults false
    *
    * @group setParam
    **/
  def setEnableScore(v: Boolean): this.type = set(enableScore, v)

  /** Path to file with list of inputs and their content, with such delimiter, readAs LINE_BY_LINE or as SPARK_DATASET. If latter is set, options is passed to spark reader.
    *
    * @group setParam
    **/
  def setSentimentDict(value: Map[String, String]): this.type = set(sentimentDict, value)

  /**
    * Tokens are needed to identify each word in a sentence boundary
    * POS tags are optionally submitted to the model in case they are needed
    * Lemmas are another optional annotator for some models
    * Bounds of sentiment are hardcoded to 0 as they render useless
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokenizedSentences = TokenizedWithSentence.unpack(annotations)

    val score = model.score(tokenizedSentences.toArray)

    Seq(Annotation(
      outputAnnotatorType,
      0,
      0,
      { if ($(enableScore)) score.toString else if (score >= 0) "positive" else "negative"},
      Map.empty[String, String]
    ))
  }

}
object SentimentDetectorModel extends ParamsAndFeaturesReadable[SentimentDetectorModel]