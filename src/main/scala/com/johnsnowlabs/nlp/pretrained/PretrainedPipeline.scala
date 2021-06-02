package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.nlp.LightPipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

/**
  * Represents a fully constructed and trained Spark NLP pipeline, ready to be used. This way, a whole pipeline can be
  * defined in 1 line. Additionally, the [[LightPipeline]] version of the model can be retrieved with member
  * `lightModel`.
  *
  * For more extended examples see the [[https://nlp.johnsnowlabs.com/docs/en/pipelines Pipelines page]] and our
  * [[https://github.com/JohnSnowLabs/spark-nlp-models Github Model Repository]] for available pipeline models.
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
  * import com.johnsnowlabs.nlp.SparkNLP
  * val testData = spark.createDataFrame(Seq(
  * (1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
  * (2, "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States")
  * )).toDF("id", "text")
  *
  * val pipeline = PretrainedPipeline("explain_document_dl", lang="en")
  *
  * val annotation = pipeline.transform(testData)
  *
  * annotation.select("entities.result").show(false)
  *
  * /*
  * +----------------------------------+
  * |result                            |
  * +----------------------------------+
  * |[Google, TensorFlow]              |
  * |[Donald John Trump, United States]|
  * +----------------------------------+
  * */
  * }}}
  *
  * @param downloadName Name of the Pipeline Model
  * @param lang Language of the defined pipeline (Default: "en")
  * @param source Source where to get the Pipeline Model
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