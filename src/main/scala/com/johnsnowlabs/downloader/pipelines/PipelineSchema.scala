package com.johnsnowlabs.downloader.pipelines

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, Dataset}

case class NLPBasic(
                     text: String,
                     document: Seq[String],
                     tokens: Seq[String],
                     normalized: Seq[String],
                     lemmas: Seq[String],
                     pos: Seq[String]
                   )

case class NLPAdvanced(
                        text: String,
                        document: Seq[String],
                        tokens: Seq[String],
                        normalized: Seq[String],
                        spelled: Seq[String],
                        stems: Seq[String],
                        lemmas: Seq[String],
                        pos: Seq[String],
                        entities: Seq[String]
                      )

trait NLPBase[T] {

  def annotate(dataset: DataFrame, inputColumn: String): Dataset[T]

  def annotate(target: String): T

  def annotate(target: Array[String]): Array[T]

  def retrieve(): PipelineModel

}