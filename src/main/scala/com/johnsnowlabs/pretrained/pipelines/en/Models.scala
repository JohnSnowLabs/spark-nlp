package com.johnsnowlabs.pretrained.pipelines.en

import com.johnsnowlabs.pretrained.pipelines.PretrainedPipeline

case class BasicPipeline() extends PretrainedPipeline("pipeline_basic", Some("en"))

case class AdvancedPipeline() extends PretrainedPipeline("pipeline_advanced", Some("en"))

case class SentimentPipeline() extends PretrainedPipeline("pipeline_vivekn", Some("en"))