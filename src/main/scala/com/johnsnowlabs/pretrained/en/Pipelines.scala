package com.johnsnowlabs.pretrained.en.pipelines

import com.johnsnowlabs.pretrained.en.models
import com.johnsnowlabs.util.PipelineModels



object SentenceDetector {
  lazy val std = PipelineModels(
    models.DocumentAssembler.std,
    models.SentenceDetector.std
  )
}

object Tokenizer {
  lazy val std = PipelineModels(
    models.DocumentAssembler.std,
    models.SentenceDetector.std,
    models.Tokenizer.std)
}

object Pos {
  lazy val fast = PipelineModels(
    models.DocumentAssembler.std,
    models.SentenceDetector.std,
    models.Tokenizer.std,
    models.Pos.fast
  )
}

object Ner {
  lazy val fast = PipelineModels(
    models.DocumentAssembler.std,
    models.SentenceDetector.std,
    models.Tokenizer.std,
    models.Pos.fast,
    models.Ner.fast
  )
}

