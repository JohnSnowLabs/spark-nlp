package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.util.{DefaultParamsReadable, MLReader}

trait HasPretrained[M <: PipelineStage] {

  /** Only MLReader types can use this interface */
  this: { def read: MLReader[M] } =>

  val defaultModelName: Option[String]

  val defaultLang: String = "en"

  lazy val defaultLoc: String = ResourceDownloader.publicLoc

  implicit private val companion = this.asInstanceOf[DefaultParamsReadable[M]]

  private val errorMsg = s"${this.getClass.getName} does not have a default pretrained model. Please provide a model name."

  /** Java default argument interoperability */

  def pretrained(name: String, lang: String, remoteLoc: String): M = {
    if (Option(name).isEmpty)
      throw new NotImplementedError(errorMsg)
    ResourceDownloader.downloadModel(companion, name, Option(lang), remoteLoc)
  }

  def pretrained(name: String, lang: String): M = pretrained(name, lang, defaultLoc)

  def pretrained(name: String): M = pretrained(name, defaultLang, defaultLoc)

  def pretrained(): M = pretrained(defaultModelName.getOrElse(throw new Exception(errorMsg)), defaultLang, defaultLoc)

}