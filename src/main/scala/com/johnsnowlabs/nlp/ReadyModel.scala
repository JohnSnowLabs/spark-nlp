package com.johnsnowlabs.nlp

import com.johnsnowlabs.downloader.ResourceDownloader
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, Dataset}

case class NLPBasic(
                     text: String,
                     tokens: Seq[String],
                     normalized: Seq[String],
                     lemmas: Seq[String],
                     pos: Seq[String]
                   )

case class NLPAdvanced(
                        text: String,
                        tokens: Seq[String],
                        normalized: Seq[String],
                        spelled: Seq[String],
                        stems: Seq[String],
                        lemmas: Seq[String],
                        pos: Seq[String],
                        entities: Seq[String]
                      )

object ReadyModel {

  trait NLPBase {

    def basic(dataset: DataFrame, inputColumn: String): Dataset[NLPBasic]

    def basic(target: String): NLPBasic

    def basic(target: Array[String]): Array[NLPBasic]

    def advanced(dataset: DataFrame, inputColumn: String): Dataset[NLPAdvanced]

    def advanced(target: String): NLPAdvanced

    def advanced(target: Array[String]): Array[NLPAdvanced]

    def spellcheck(dataset: DataFrame, inputColumn: String): Dataset[String]

    def spellcheck(target: String): String

    def spellcheck(target: Array[String]): Array[String]

    def sentiment(dataset: DataFrame, inputColumn: String): Dataset[String]

    def sentiment(target: String): String

    def sentiment(target: Array[String]): Array[String]

  }

  object en extends NLPBase {

    import ResourceHelper.spark.implicits._

    lazy private val pipelineBasic: PipelineModel = ResourceDownloader
      .downloadPipeline("pipeline_basic", Some("en"))

    lazy private val pipelineAdvanced: PipelineModel = ResourceDownloader
      .downloadPipeline("pipeline_advanced", Some("en"))

    override def basic(dataset: DataFrame, inputColumn: String): Dataset[NLPBasic] = {
      pipelineBasic
        .transform(dataset.withColumnRenamed(inputColumn, "text"))
        .select(inputColumn, "finished_token", "finished_normal", "finished_lemma", "finished_pos")
        .toDF("text", "tokens", "normalized", "lemmas", "pos")
        .as[NLPBasic]
    }

    override def basic(target: String): NLPBasic = {
      val extracted = new LightPipeline(pipelineBasic).annotate(target)
      NLPBasic(
        text = target,
        tokens = extracted("token"),
        normalized = extracted("normal"),
        lemmas = extracted("lemma"),
        pos = extracted("pos")
      )
    }

    override def basic(target: Array[String]): Array[NLPBasic] = {
      val allExtracted = new LightPipeline(pipelineBasic).annotate(target)
      target.zip(allExtracted).map{ case (doc, extracted) => {
        NLPBasic(
          text = doc,
          tokens = extracted("token"),
          normalized = extracted("normal"),
          lemmas = extracted("lemma"),
          pos = extracted("pos")
        )
      }}
    }

    override def advanced(dataset: DataFrame, inputColumn: String): Dataset[NLPAdvanced] = {
      pipelineAdvanced
        .transform(dataset.withColumnRenamed(inputColumn, "text"))
        .select(inputColumn, "finished_token", "finished_normal", "finished_spell", "finished_stem", "finished_lemma", "finished_pos", "finished_ner")
        .toDF("text", "tokens", "normalized", "spelled", "stems", "lemmas", "pos", "entities")
        .as[NLPAdvanced]
    }

    override def advanced(target: String): NLPAdvanced = {
      val extracted = new LightPipeline(pipelineAdvanced).annotate(target)
      NLPAdvanced(
        text = target,
        tokens = extracted("token"),
        normalized = extracted("normal"),
        spelled = extracted("spell"),
        stems = extracted("stem"),
        lemmas = extracted("lemma"),
        pos = extracted("pos"),
        entities = extracted("ner")
      )
    }

    override def advanced(target: Array[String]): Array[NLPAdvanced] = {
      val allExtracted = new LightPipeline(pipelineAdvanced).annotate(target)
      target.zip(allExtracted).map{case (doc, extracted) => {
        NLPAdvanced(
          text = doc,
          tokens = extracted("token"),
          normalized = extracted("normal"),
          spelled = extracted("spell"),
          stems = extracted("stem"),
          lemmas = extracted("lemma"),
          pos = extracted("pos"),
          entities = extracted("ner")
        )
      }}    }

    override def spellcheck(dataset: DataFrame, inputColumn: String): Dataset[String] = ???

    override def spellcheck(target: String): String = ???

    override def spellcheck(target: Array[String]): Array[String] = ???

    override def sentiment(dataset: DataFrame, inputColumn: String): Dataset[String] = ???

    override def sentiment(target: String): String = ???

    override def sentiment(target: Array[String]): Array[String] = ???
  }

}
