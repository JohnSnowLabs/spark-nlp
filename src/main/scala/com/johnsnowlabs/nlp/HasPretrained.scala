package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.util.{DefaultParamsReadable, MLReader}

trait HasPretrained[M <: PipelineStage] {

  /** Only MLReader types can use this interface */
  this: { def read: MLReader[M] } =>

  val defaultModelName: String

  val defaultLang: String = "en"

  val defaultLoc: String = ResourceDownloader.publicLoc

  implicit private val companion = this.asInstanceOf[DefaultParamsReadable[M]]

  /** Java default argument interoperability */

  def pretrained(name: String, lang: String, remoteLoc: String): M = {
    if (Option(name).isEmpty)
      throw new NotImplementedError(s"${this.getClass.getName} does not have a default pretrained model. Please provide a model name.")
    ResourceDownloader.downloadModel(companion, name, Option(lang), remoteLoc)
  }

  def pretrained(name: String, lang: String): M = pretrained(name, lang, defaultLoc)

  def pretrained(name: String): M = pretrained(name, defaultLang, defaultLoc)

  def pretrained(): M = pretrained(defaultModelName, defaultLang, defaultLoc)

}