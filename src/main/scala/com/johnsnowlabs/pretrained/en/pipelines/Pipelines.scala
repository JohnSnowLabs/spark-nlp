package com.johnsnowlabs.pretrained.en.pipelines

import com.johnsnowlabs.pretrained.en.models._
import com.johnsnowlabs.util.PipelineModels

object S3POSPipeline {
  def retrieveSmall = PipelineModels(
    S3DocumentAssembler.retrieveStandard,
    S3SentenceDetector.retrieveStandard,
    S3Tokenizer.retrieveStandard,
    S3PerceptronModel.retrieveSmall
  )
}

object S3NerCrfPipeline {
  def retrieveSmall = PipelineModels(
    S3DocumentAssembler.retrieveStandard,
    S3SentenceDetector.retrieveStandard,
    S3Tokenizer.retrieveStandard,
    S3PerceptronModel.retrieveSmall,
    S3NerCrfModel.retrieveSmall
  )
}
