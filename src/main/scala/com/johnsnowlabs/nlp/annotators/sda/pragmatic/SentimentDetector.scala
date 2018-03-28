package com.johnsnowlabs.nlp.annotators.sda.pragmatic

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, SENTIMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class SentimentDetector(override val uid: String) extends AnnotatorApproach[SentimentDetectorModel] {

  override val annotatorType: AnnotatorType = SENTIMENT

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  override val description: String = "Rule based sentiment detector"

  val dictionary = new ExternalResourceParam(this, "dictionary", "delimited file with a list sentiment tags per word. Requires 'delimiter' in options")

  def this() = this(Identifiable.randomUID("SENTIMENT"))

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
      .setSentimentDict(ResourceHelper.parseKeyValueText($(dictionary)))
  }

}
object SentimentDetector extends DefaultParamsReadable[SentimentDetector]
