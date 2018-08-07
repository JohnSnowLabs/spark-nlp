package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class DeIdentification(override val uid: String) extends AnnotatorApproach[DeIdentificationModel]{

  override val description: String = "Protect personal information on documents"
  override val annotatorType: AnnotatorType = DOCUMENT
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, TOKEN, CHUNK)

  val regexPatternsDictionary = new ExternalResourceParam(this, "regexPatternsDictionary",
  "dictionary with regular expression patterns that match some protected entity")

  def this() = this(Identifiable.randomUID("DE-IDENTIFICATION"))

  def setRegexPatternsDictionary(value: ExternalResource): this.type = {
    set(regexPatternsDictionary, value)
  }

  def transformRegexPatternsDictionary(regexPatternsDictionary: List[(String, String)]):
  Map[String, List[String]] = {

    if (regexPatternsDictionary.isEmpty){
      return Map()
    }

    regexPatternsDictionary.groupBy(_._1) //group by entity
      .mapValues(regexDicList => regexDicList.map(regexDic => regexDic._2))
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DeIdentificationModel = {
    val regexPatternDictionary:List[(String, String)] = if (get(regexPatternsDictionary).isDefined)
      ResourceHelper.parseTupleText($(regexPatternsDictionary)).toList
    else
      List()

    val dictionary = transformRegexPatternsDictionary(regexPatternDictionary)

    new DeIdentificationModel()
      .setRegexPatternsDictionary(dictionary)
  }

}

object DeIdentification extends DefaultParamsReadable[DeIdentification]