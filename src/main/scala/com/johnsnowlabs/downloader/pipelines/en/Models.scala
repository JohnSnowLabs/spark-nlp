package com.johnsnowlabs.downloader.pipelines.en

import com.johnsnowlabs.downloader.ResourceDownloader
import com.johnsnowlabs.downloader.pipelines.{NLPAdvanced, NLPBase, NLPBasic}
import com.johnsnowlabs.nlp.LightPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, Dataset}

import ResourceHelper.spark.implicits._

object BasicPipeline extends NLPBase[NLPBasic] {

  lazy private val pipelineBasic: PipelineModel = ResourceDownloader
    .downloadPipeline("pipeline_basic_fin", Some("en"))

  lazy private val pipelineBasicAnn: PipelineModel = ResourceDownloader
    .downloadPipeline("pipeline_basic_ann", Some("en"))

  override def annotate(dataset: DataFrame, inputColumn: String): Dataset[NLPBasic] = {
    pipelineBasic
      .transform(dataset.withColumnRenamed(inputColumn, "text"))
      .select(inputColumn, "finished_document", "finished_token", "finished_normal", "finished_lemma", "finished_pos")
      .toDF("text", "document", "tokens", "normalized", "lemmas", "pos")
      .as[NLPBasic]
  }

  override def annotate(target: String): NLPBasic = {
    val extracted = new LightPipeline(pipelineBasicAnn).annotate(target)
    NLPBasic(
      text = target,
      document = extracted("document"),
      tokens = extracted("token"),
      normalized = extracted("normal"),
      lemmas = extracted("lemma"),
      pos = extracted("pos")
    )
  }

  override def annotate(target: Array[String]): Array[NLPBasic] = {
    val allExtracted = new LightPipeline(pipelineBasicAnn).annotate(target)
    target.zip(allExtracted).map { case (doc, extracted) =>
      NLPBasic(
        text = doc,
        document = extracted("document"),
        tokens = extracted("token"),
        normalized = extracted("normal"),
        lemmas = extracted("lemma"),
        pos = extracted("pos")
      )
    }
  }

  override def retrieve(): PipelineModel = {
    pipelineBasicAnn
  }

}

object AdvancedPipeline extends NLPBase[NLPAdvanced] {

  lazy private val pipelineAdvanced: PipelineModel = ResourceDownloader
    .downloadPipeline("pipeline_advanced_fin", Some("en"))

  lazy private val pipelineAdvancedAnn: PipelineModel = ResourceDownloader
    .downloadPipeline("pipeline_advanced_ann", Some("en"))

  override def annotate(dataset: DataFrame, inputColumn: String): Dataset[NLPAdvanced] = {
    pipelineAdvanced
      .transform(dataset.withColumnRenamed(inputColumn, "text"))
      .select(inputColumn, "finished_document", "finished_token", "finished_normal", "finished_spell", "finished_stem", "finished_lemma", "finished_pos", "finished_ner")
      .toDF("text", "document", "tokens", "normalized", "spelled", "stems", "lemmas", "pos", "entities")
      .as[NLPAdvanced]
  }

  override def annotate(target: String): NLPAdvanced = {
    val extracted = new LightPipeline(pipelineAdvancedAnn).annotate(target)
    NLPAdvanced(
      text = target,
      document = extracted("document"),
      tokens = extracted("token"),
      normalized = extracted("normal"),
      spelled = extracted("spell"),
      stems = extracted("stem"),
      lemmas = extracted("lemma"),
      pos = extracted("pos"),
      entities = extracted("ner")
    )
  }

  override def annotate(target: Array[String]): Array[NLPAdvanced] = {
    val allExtracted = new LightPipeline(pipelineAdvancedAnn).annotate(target)
    target.zip(allExtracted).map { case (doc, extracted) =>
      NLPAdvanced(
        text = doc,
        document = extracted("document"),
        tokens = extracted("token"),
        normalized = extracted("normal"),
        spelled = extracted("spell"),
        stems = extracted("stem"),
        lemmas = extracted("lemma"),
        pos = extracted("pos"),
        entities = extracted("ner")
      )
    }
  }

  override def retrieve(): PipelineModel = {
    pipelineAdvancedAnn
  }

}
