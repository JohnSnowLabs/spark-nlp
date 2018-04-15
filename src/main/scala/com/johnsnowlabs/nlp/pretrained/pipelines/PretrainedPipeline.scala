package com.johnsnowlabs.nlp.pretrained.pipelines

import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.{Finisher, LightPipeline}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

abstract class PretrainedPipeline(downloadName: String, language: Option[String] = None, folder: String = ResourceDownloader.publicFolder) {

  lazy protected val modelCache: PipelineModel = ResourceDownloader
    .downloadPipeline(downloadName, language, folder)

  def annotate(dataset: DataFrame, inputColumn: String): DataFrame = {
    modelCache
      .transform(dataset.withColumnRenamed(inputColumn, "text"))
  }

  def annotate(target: String): Map[String, Seq[String]] = new LightPipeline(modelCache).annotate(target)

  def annotate(target: Array[String]): Array[Map[String, Seq[String]]] = new LightPipeline(modelCache).annotate(target)

  def pretrained(): PipelineModel = modelCache

}