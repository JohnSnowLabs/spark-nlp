package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import com.typesafe.config.Config
import com.johnsnowlabs.nlp.util.ConfigHelper
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

import scala.collection.JavaConverters._

/**
  * Created by saif on 28/04/17.
  */

/**
  * Class to find standarized lemmas from words. Uses a user-provided or default dictionary.
  * @param uid required internal uid provided by constructor
  * @@ lemmaDict: A dictionary of predefined lemmas must be provided
  */
class Lemmatizer(override val uid: String) extends AnnotatorModel[Lemmatizer] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  private val config: Config = ConfigHelper.retrieve

  val lemmaDict: MapFeature[String, String] = new MapFeature(this, "lemmaDict", "provide a lemma dictionary")

  val lemmaFormat: Param[String] = new Param[String](this, "lemmaFormat", "TXT or TXTDS for reading dictionary as dataset")

  val lemmaKeySep: Param[String] = new Param[String](this, "lemmaKeySep", "lemma dictionary key separator")

  val lemmaValSep: Param[String] = new Param[String](this, "lemmaValSep", "lemma dictionary value separator")

  setDefault(lemmaFormat, config.getString("nlp.lemmaDict.format"))

  setDefault(lemmaKeySep, config.getString("nlp.lemmaDict.kvSeparator"))

  setDefault(lemmaValSep, config.getString("nlp.lemmaDict.vSeparator"))

  if (config.getString("nlp.lemmaDict.file").nonEmpty)
    set(lemmaDict,  Lemmatizer.retrieveLemmaDict(
      config.getString("nlp.lemmaDict.file"),
      config.getString("nlp.lemmaDict.format"),
      config.getString("nlp.lemmaDict.kvSeparator"),
      config.getString("nlp.lemmaDict.vSeparator")))

  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("LEMMATIZER"))

  def getLemmaDict: Map[String, String] = $$(lemmaDict)
  protected def getLemmaFormat: String = $(lemmaFormat)
  protected def getLemmaKeySep: String = $(lemmaKeySep)
  protected def getLemmaValSep: String = $(lemmaValSep)

  def setLemmaDict(dictionary: String): this.type = {
    set(lemmaDict, Lemmatizer.retrieveLemmaDict(dictionary, $(lemmaFormat), $(lemmaKeySep), $(lemmaValSep)))
  }
  def setLemmaDictHMap(dictionary: java.util.HashMap[String, String]): this.type = {
    set(lemmaDict, dictionary.asScala.toMap)
  }
  def setLemmaDictMap(dictionary: Map[String, String]): this.type = {
    set(lemmaDict, dictionary)
  }
  def setLemmaFormat(value: String): this.type = set(lemmaFormat, value)
  def setLemmaKeySep(value: String): this.type = set(lemmaKeySep, value)
  def setLemmaValSep(value: String): this.type = set(lemmaValSep, value)

  /**
    * @return one to one annotation from token to a lemmatized word, if found on dictionary or leave the word as is
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { tokenAnnotation =>
      val token = tokenAnnotation.result
      Annotation(
        annotatorType,
        tokenAnnotation.begin,
        tokenAnnotation.end,
        $$(lemmaDict).getOrElse(token, token),
        tokenAnnotation.metadata
      )
    }
  }
}

object Lemmatizer extends ParamsAndFeaturesReadable[Lemmatizer] {
  protected def retrieveLemmaDict(
                         lemmaFilePath: String,
                         lemmaFormat: String,
                         lemmaKeySep: String,
                         lemmaValSep: String
                       ): Map[String, String] = {
    ResourceHelper.flattenRevertValuesAsKeys(lemmaFilePath, lemmaFormat.toUpperCase, lemmaKeySep, lemmaValSep)
  }
}
