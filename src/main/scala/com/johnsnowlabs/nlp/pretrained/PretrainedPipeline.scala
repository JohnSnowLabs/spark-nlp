package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.nlp.LightPipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

case class PretrainedPipeline(
                               downloadName: String,
                               lang: String = "en",
                               source: String = ResourceDownloader.publicLoc
                             ) {

  val model: PipelineModel = ResourceDownloader
    .downloadPipeline(downloadName, Option(lang), source)

  lazy val lightModel = new LightPipeline(model)

  def annotate(dataset: DataFrame, inputColumn: String): DataFrame = {
    model
      .transform(dataset.withColumnRenamed(inputColumn, "text"))
  }

  def annotate(target: String): Map[String, Seq[String]] = lightModel.annotate(target)

  def annotate(target: Array[String]): Array[Map[String, Seq[String]]] = lightModel.annotate(target)

  def transform(dataFrame: DataFrame): DataFrame = model.transform(dataFrame)

}