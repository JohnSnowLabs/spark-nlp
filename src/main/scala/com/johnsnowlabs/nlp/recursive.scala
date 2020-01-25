package com.johnsnowlabs.nlp

import org.apache.spark.ml.{Pipeline, PipelineModel}

package object recursive {

  implicit def p2recursive(pipeline: Pipeline): RecursivePipeline =
    new RecursivePipeline(pipeline)
  implicit def pm2recursive(pipelineModel: PipelineModel): RecursivePipelineModel =
    new RecursivePipelineModel(pipelineModel.uid, pipelineModel)
  implicit def pm2light(pipelineModel: PipelineModel): LightPipeline =
    new LightPipeline(pipelineModel)

  implicit class Recursive(p: Pipeline) {
    def recursive: RecursivePipeline = {
      new RecursivePipeline(p)
    }
  }

  implicit class RecursiveModel(p: PipelineModel) {
    def recursive: RecursivePipelineModel = {
      new RecursivePipelineModel(p.uid, p)
    }
  }

}
