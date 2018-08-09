package com.johnsnowlabs.nlp.annotators.sda.pragmatic

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, SENTIMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{DoubleParam, IntParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class SentimentDetector(override val uid: String) extends AnnotatorApproach[SentimentDetectorModel] {

  override val annotatorType: AnnotatorType = SENTIMENT

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  override val description: String = "Rule based sentiment detector"

  def this() = this(Identifiable.randomUID("SENTIMENT"))

  val positiveMultiplier = new DoubleParam(this, "positiveMultiplier", "multiplier for positive sentiments. Defaults 1.0")
  val negativeMultiplier = new DoubleParam(this, "negativeMultiplier", "multiplier for negative sentiments. Defaults -1.0")
  val incrementMultiplier = new DoubleParam(this, "incrementMultiplier", "multiplier for increment sentiments. Defaults 2.0")
  val decrementMultiplier = new IntParam(this, "decrementMultiplier", "multiplier for decrement sentiments. Defaults -2.0")
  val reverseMultiplier = new DoubleParam(this, "reverseMultiplier", "multiplier for revert sentiments. Defaults -1.0")

  val dictionary = new ExternalResourceParam(this, "dictionary", "delimited file with a list sentiment tags per word. Requires 'delimiter' in options")

  setDefault(
    positiveMultiplier -> 1.0,
    negativeMultiplier -> -1.0,
    incrementMultiplier -> 2.0,
    decrementMultiplier -> -2,
    reverseMultiplier -> -1.0
  )

  def setPositiveMultipler(v: Double): this.type = set(positiveMultiplier, v)
  def setNegativeMultipler(v: Double): this.type = set(negativeMultiplier, v)
  def setIncrementMultipler(v: Double): this.type = set(incrementMultiplier, v)
  def setDecrementMultipler(v: Int): this.type = set(decrementMultiplier, v)
  def setReverseMultipler(v: Double): this.type = set(reverseMultiplier, v)

  def setDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "dictionary needs 'delimiter' in order to separate words from sentiment tags")
    set(dictionary, value)
  }

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
      .setSentimentDict(ResourceHelper.parseKeyValueText($(dictionary)))
  }

}
object SentimentDetector extends DefaultParamsReadable[SentimentDetector]
