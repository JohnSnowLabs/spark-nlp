package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}


object PipelineModels {

  lazy val dummyDataset = {
    import ResourceHelper.spark.implicits._
    ResourceHelper.spark.createDataset(Seq.empty[String]).toDF("text")
  }

  def apply(stages: Transformer*): PipelineModel = {
    new Pipeline().setStages(stages.toArray).fit(dummyDataset)
  }
}
