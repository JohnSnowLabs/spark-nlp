package com.johnsnowlabs.nlp

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.DefaultParamsReadable

object base {

  type DocumentAssembler = com.johnsnowlabs.nlp.DocumentAssembler
  object DocumentAssembler extends DefaultParamsReadable[DocumentAssembler]

  type ChunkAssembler = com.johnsnowlabs.nlp.ChunkAssembler
  object ChunkAssembler extends DefaultParamsReadable[ChunkAssembler]

  type TokenAssembler = com.johnsnowlabs.nlp.TokenAssembler
  object TokenAssembler extends DefaultParamsReadable[TokenAssembler]

  type Finisher = com.johnsnowlabs.nlp.Finisher
  object Finisher extends DefaultParamsReadable[Finisher]

  type RecursivePipeline = com.johnsnowlabs.nlp.RecursivePipeline

  type LightPipeline = com.johnsnowlabs.nlp.LightPipeline

  implicit def pip2sparkless(pipelineModel: PipelineModel): LightPipeline = {
    LightPipeline.pip2sparkless(pipelineModel)
  }

}
