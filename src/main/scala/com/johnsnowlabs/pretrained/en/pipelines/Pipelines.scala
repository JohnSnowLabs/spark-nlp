package com.johnsnowlabs.pretrained.en.pipelines

import com.johnsnowlabs.pretrained.en.models._
import com.johnsnowlabs.util.PipelineModels

object CloudPOSPipeline {
  def retrieveSmall = PipelineModels(
    CloudDocumentAssembler.retrieveStandard,
    CloudSentenceDetector.retrieveStandard,
    CloudTokenizer.retrieveStandard,
    CloudPerceptronModel.retrieveSmall
  )
}

object CloudNerCrfPipeline {
  def retrieveSmall = PipelineModels(
    CloudDocumentAssembler.retrieveStandard,
    CloudSentenceDetector.retrieveStandard,
    CloudTokenizer.retrieveStandard,
    CloudPerceptronModel.retrieveSmall,
    CloudNerCrfModel.retrieveSmall
  )
}
