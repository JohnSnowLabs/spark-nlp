package com.johnsnowlabs.pretrained.pipelines.en

import com.johnsnowlabs.pretrained.pipelines.PretrainedPipeline

class BasicPipeline extends PretrainedPipeline("pipeline_basic", Some("en")) {
  override val columns = Array("text", "document", "token", "normal", "lemma", "pos")
}

class AdvancedPipeline extends PretrainedPipeline("pipeline_advanced", Some("en")) {
  override val columns = Array("text", "document", "tokens", "normalized", "spelled", "stems", "lemmas", "pos", "entities")
}

class SentimentPipeline extends PretrainedPipeline("pipeline_vivekn", Some("en")) {
  override protected val columns: Array[String] = Array("text", "document", "token", "normal", "spell", "sentiment")
}