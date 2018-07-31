package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class DeIdentification(override val uid: String) extends AnnotatorApproach[DeIdentificationModel]{

  override val description: String = "Protect personal information on documents"
  override val annotatorType: AnnotatorType = DOCUMENT
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, CHUNK)

  val regexPatternsDictionary = new ExternalResourceParam(this, "regexPatternsDictionary",
  "dictionary with regular expression patterns that match some protected entity")

  def this() = this(Identifiable.randomUID("DE-IDENTIFICATION"))

  def setRegexPatternsDictionary(path: String,
                                 delimiter: String,
                                 readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                                 options: Map[String, String] = Map("format" -> "text")): this.type = {
    set(regexPatternsDictionary, ExternalResource(path, readAs, options))
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DeIdentificationModel = {
    val regexPatternDictionary = if (get(regexPatternsDictionary).isDefined)
      ResourceHelper.parseKeyValueText($(regexPatternsDictionary))
    else
      Map.empty[String, String]

    new DeIdentificationModel()
      .setRegexPatternsDictionary(regexPatternDictionary) //set the processed version Map(Entity->List[String])
  }

}

object DeIdentification extends DefaultParamsReadable[DeIdentification]