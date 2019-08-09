package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ResourceHelper, ReadAs}
import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/**
  * Class to find standarized lemmas from words. Uses a user-provided or default dictionary.
  * @param uid required internal uid provided by constructor
  * @@ lemmaDict: A dictionary of predefined lemmas must be provided
  */
class Lemmatizer(override val uid: String) extends AnnotatorApproach[LemmatizerModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val description: String = "Retrieves the significant part of a word"

  val dictionary: ExternalResourceParam = new ExternalResourceParam(this, "dictionary", "lemmatizer external dictionary." +
    " needs 'keyDelimiter' and 'valueDelimiter' in options for parsing target text")

  override val outputAnnotatorType: AnnotatorType = TOKEN

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("LEMMATIZER"))

  def getDictionary: ExternalResource = $(dictionary)

  def setDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("keyDelimiter") && value.options.contains("valueDelimiter"),
      "Lemmatizer dictionary requires options with 'keyDelimiter' and 'valueDelimiter'")
    set(dictionary, value)
  }

  def setDictionary(
                     path: String,
                     keyDelimiter: String,
                     valueDelimiter: String,
                     readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                     options: Map[String, String] = Map("format" -> "text")): this.type =
    set(dictionary, ExternalResource(path, readAs, options ++ Map("keyDelimiter" -> keyDelimiter, "valueDelimiter" -> valueDelimiter)))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): LemmatizerModel = {
    new LemmatizerModel()
      .setLemmaDict(ResourceHelper.flattenRevertValuesAsKeys($(dictionary)))
  }

}

object Lemmatizer extends DefaultParamsReadable[Lemmatizer]
