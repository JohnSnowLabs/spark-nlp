package com.johnsnowlabs.nlp.pretrained.pipelines.en

import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.pretrained.pipelines.PretrainedPipeline

case class BasicPipeline() extends PretrainedPipeline("pipeline_basic", ResourceDownloader.publicFolder, Some("en"))

case class AdvancedPipeline() extends PretrainedPipeline("pipeline_advanced", ResourceDownloader.publicFolder, Some("en"))

case class SentimentPipeline() extends PretrainedPipeline("pipeline_vivekn", ResourceDownloader.publicFolder, Some("en"))