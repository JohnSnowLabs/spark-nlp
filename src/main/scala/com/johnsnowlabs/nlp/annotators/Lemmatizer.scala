package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.AnnotatorApproach
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/**
  * Created by saif on 28/04/17.
  */

/**
  * Class to find standarized lemmas from words. Uses a user-provided or default dictionary.
  * @param uid required internal uid provided by constructor
  * @@ lemmaDict: A dictionary of predefined lemmas must be provided
  */
class Lemmatizer(override val uid: String) extends AnnotatorApproach[LemmatizerModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val description: String = "Retrieves the significant part of a word"

  val lemmaDictPath: Param[String] = new Param[String](this, "lemmaDictPath", "path to lemma dictionary")

  val lemmaFormat: Param[String] = new Param[String](this, "lemmaFormat", "TXT or TXTDS for reading dictionary as dataset")

  val lemmaKeySep: Param[String] = new Param[String](this, "lemmaKeySep", "lemma dictionary key separator")

  val lemmaValSep: Param[String] = new Param[String](this, "lemmaValSep", "lemma dictionary value separator")

  setDefault(
    lemmaDictPath -> "/lemma-corpus/AntBNC_lemmas_ver_001.txt",
    lemmaFormat -> "TXT",
    lemmaKeySep -> "->",
    lemmaValSep -> "\t"
  )

  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("LEMMATIZER"))

  def getLemmaDictPath: String = $(lemmaDictPath)
  def getLemmaFormat: String = $(lemmaFormat)
  def getLemmaKeySep: String = $(lemmaKeySep)
  def getLemmaValSep: String = $(lemmaValSep)

  def setLemmaDictPath(value: String): this.type = set(lemmaDictPath, value)
  def setLemmaFormat(value: String): this.type = set(lemmaFormat, value)
  def setLemmaKeySep(value: String): this.type = set(lemmaKeySep, value)
  def setLemmaValSep(value: String): this.type = set(lemmaValSep, value)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): LemmatizerModel = {
    new LemmatizerModel()
      .setLemmaDict(ResourceHelper.flattenRevertValuesAsKeys(
        $(lemmaDictPath),
        $(lemmaFormat).toUpperCase,
        $(lemmaKeySep),
        $(lemmaValSep))
      )
  }

}

object Lemmatizer extends DefaultParamsReadable[Lemmatizer]
