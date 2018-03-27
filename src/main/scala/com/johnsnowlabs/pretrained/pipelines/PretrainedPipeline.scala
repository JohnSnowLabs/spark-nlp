package com.johnsnowlabs.pretrained.pipelines

import com.johnsnowlabs.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.{Finisher, LightPipeline}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

abstract class PretrainedPipeline(downloadName: String, language: Option[String]) {

  lazy protected val modelCache: PipelineModel = ResourceDownloader
    .downloadPipeline(downloadName, language)

  protected val columns: Array[String]

  def annotate(dataset: DataFrame, inputColumn: String, useFinisher: Boolean = true): DataFrame = {
    val result = modelCache
      .transform(dataset.withColumnRenamed(inputColumn, "text"))
      .select(columns.head, columns.tail:_*)
    if (useFinisher)
      new Finisher().setInputCols(columns)
        .transform(result)
    else
      result
  }

  def annotate(target: String): Map[String, Seq[String]] = new LightPipeline(modelCache).annotate(target)

  def annotate(target: Array[String]): Array[Map[String, Seq[String]]] = new LightPipeline(modelCache).annotate(target)

  def retrieve(): PipelineModel = modelCache

}