package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.common.StringMapParam
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.typesafe.config.Config
import com.johnsnowlabs.nlp.util.ConfigHelper
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

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

  val lemmaDict: StringMapParam = new StringMapParam(this, "lemmaDict", "provide a lemma dictionary")

  val lemmaFormat: Param[String] = new Param[String](this, "lemmaFormat", "TXT or TXTDS for reading dictionary as dataset")

  val lemmaKeySep: Param[String] = new Param[String](this, "lemmaKeySep", "lemma dictionary key separator")

  val lemmaValSep: Param[String] = new Param[String](this, "lemmaValSep", "lemma dictionary value separator")

  setDefault(lemmaFormat, config.getString("nlp.lemmaDict.format"))

  setDefault(lemmaKeySep, config.getString("nlp.lemmaDict.kvSeparator"))

  setDefault(lemmaValSep, config.getString("nlp.lemmaDict.vSeparator"))

  if (config.getString("nlp.lemmaDict.file").nonEmpty)
    setDefault(lemmaDict,  Lemmatizer.retrieveLemmaDict(
      config.getString("nlp.lemmaDict.file"),
      config.getString("nlp.lemmaDict.format"),
      config.getString("nlp.lemmaDict.kvSeparator"),
      config.getString("nlp.lemmaDict.vSeparator")))

  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("LEMMATIZER"))

  def getLemmaDict: Map[String, String] = $(lemmaDict)

  def setLemmaDict(dictionary: String): this.type = {
    set(lemmaDict, Lemmatizer.retrieveLemmaDict(dictionary, $(lemmaFormat), $(lemmaKeySep), $(lemmaValSep)))
  }

  def setLemmaDictHMap(dictionary: java.util.HashMap[String, String]): this.type = {
    set(lemmaDict, dictionary.asScala.toMap)
  }

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
        $(lemmaDict).getOrElse(token, token),
        tokenAnnotation.metadata
      )
    }
  }
}

object Lemmatizer extends DefaultParamsReadable[Lemmatizer] {

  /**
    * Retrieves Lemma dictionary from configured compiled source set in configuration
    * @return a Dictionary for lemmas
    */
  protected def retrieveLemmaDict(
                         lemmaFilePath: String,
                         lemmaFormat: String,
                         lemmaKeySep: String,
                         lemmaValSep: String
                       ): Map[String, String] = {
    ResourceHelper.flattenRevertValuesAsKeys(lemmaFilePath, lemmaFormat.toUpperCase, lemmaKeySep, lemmaValSep)
  }
}
