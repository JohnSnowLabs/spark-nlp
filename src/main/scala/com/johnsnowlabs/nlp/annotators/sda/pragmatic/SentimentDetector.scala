package com.johnsnowlabs.nlp.annotators.sda.pragmatic

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, SENTIMENT, TOKEN}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class SentimentDetector(override val uid: String) extends AnnotatorApproach[SentimentDetectorModel] {

  override val annotatorType: AnnotatorType = SENTIMENT

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  override val description: String = "Rule based sentiment detector"

  def this() = this(Identifiable.randomUID("SENTIMENT"))

  val dictPath = new Param[String](this, "dictPath", "path to dictionary for pragmatic sentiment analysis")

  val dictFormat = new Param[String](this, "dictFormat", "format of dictionary, can be TXT or TXTDS for read as dataset")

  val dictSeparator = new Param[String](this, "dictSeparator", "key value separator for dictionary")

  setDefault(
    dictFormat -> "TXT",
    dictSeparator -> ","
  )

  def setDictPath(path: String): this.type = set(dictPath, path)

  def getDictPath: String = $(dictPath)

  def setDictFormat(format: String): this.type = set(dictFormat, format)

  def getDictFormat: String = $(dictFormat)

  def setDictSeparator(separator: String): this.type = set(dictSeparator, separator)

  def getDictSeparator: String = $(dictSeparator)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SentimentDetectorModel = {
    new SentimentDetectorModel()
      .setSentimentDict(ResourceHelper.parseKeyValueText($(dictPath), $(dictFormat).toUpperCase, $(dictSeparator)))
  }

}
object SentimentDetector extends DefaultParamsReadable[SentimentDetector]
