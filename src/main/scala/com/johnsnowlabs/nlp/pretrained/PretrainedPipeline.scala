package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.nlp.LightPipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

/**
  * TODO
  * @param downloadName
  * @param lang
  * @param source
  * @param parseEmbeddingsVectors
  * @param diskLocation
  */
case class PretrainedPipeline(
                               downloadName: String,
                               lang: String = "en",
                               source: String = ResourceDownloader.publicLoc,
                               parseEmbeddingsVectors: Boolean = false,
                               diskLocation: Option[String] = None
                             ) {

  /** Support for java default argument interoperability */
  def this(downloadName: String) {
    this(downloadName, "en", ResourceDownloader.publicLoc)
  }

  def this(downloadName: String, lang: String) {
    this(downloadName, lang, ResourceDownloader.publicLoc)
  }

  val model: PipelineModel = if (diskLocation.isEmpty) {
    ResourceDownloader
      .downloadPipeline(downloadName, Option(lang), source)
  } else {
    PipelineModel.load(diskLocation.get)
  }

  lazy val lightModel = new LightPipeline(model, parseEmbeddingsVectors)

  def annotate(dataset: DataFrame, inputColumn: String): DataFrame = {
    model
      .transform(dataset.withColumnRenamed(inputColumn, "text"))
  }

  def annotate(target: String): Map[String, Seq[String]] = lightModel.annotate(target)

  def annotate(target: Array[String]): Array[Map[String, Seq[String]]] = lightModel.annotate(target)

  def transform(dataFrame: DataFrame): DataFrame = model.transform(dataFrame)

}

object PretrainedPipeline {
  def fromDisk(path: String, parseEmbeddings: Boolean = false): PretrainedPipeline = {
    PretrainedPipeline(null, null, null, parseEmbeddings, Some(path))
  }
  def fromDisk(path: String): PretrainedPipeline = {
    fromDisk(path, false)
  }
}