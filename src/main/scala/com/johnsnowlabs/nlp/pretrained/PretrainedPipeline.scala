/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.nlp.{IAnnotation, LightPipeline}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

/** Represents a fully constructed and trained Spark NLP pipeline, ready to be used. This way, a
  * whole pipeline can be defined in 1 line. Additionally, the [[LightPipeline]] version of the
  * model can be retrieved with member `lightModel`.
  *
  * For more extended examples see the
  * [[https://nlp.johnsnowlabs.com/docs/en/pipelines Pipelines page]] and our
  * [[https://github.com/JohnSnowLabs/spark-nlp-models Github Model Repository]] for available
  * pipeline models.
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
  * import com.johnsnowlabs.nlp.SparkNLP
  * val testData = spark.createDataFrame(Seq(
  * (1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
  * (2, "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States")
  * )).toDF("id", "text")
  *
  * val pipeline = PretrainedPipeline("explain_document_dl", lang="en")
  *
  * val annotation = pipeline.transform(testData)
  *
  * annotation.select("entities.result").show(false)
  *
  * /*
  * +----------------------------------+
  * |result                            |
  * +----------------------------------+
  * |[Google, TensorFlow]              |
  * |[Donald John Trump, United States]|
  * +----------------------------------+
  * */
  * }}}
  *
  * @param downloadName
  *   Name of the Pipeline Model
  * @param lang
  *   Language of the defined pipeline (Default: "en")
  * @param source
  *   Source where to get the Pipeline Model
  * @param parseEmbeddingsVectors
  * @param diskLocation
  */
case class PretrainedPipeline(
    downloadName: String,
    lang: String = "en",
    source: String = ResourceDownloader.publicLoc,
    parseEmbeddingsVectors: Boolean = false,
    diskLocation: Option[String] = None) {

  /** Support for java default argument interoperability */
  def this(downloadName: String) {
    this(downloadName, "en", ResourceDownloader.publicLoc)
  }

  def this(downloadName: String, lang: String) {
    this(downloadName, lang, ResourceDownloader.publicLoc)
  }

  val model: PipelineModel = if (diskLocation.isEmpty) {
    ResourceDownloader
      .downloadPipeline(downloadName, Option(lang), source)
  } else {
    PipelineModel.load(diskLocation.get)
  }

  lazy val lightModel = new LightPipeline(model, parseEmbeddingsVectors)

  def annotate(dataset: DataFrame, inputColumn: String): DataFrame = {
    model
      .transform(dataset.withColumnRenamed(inputColumn, "text"))
  }

  def annotate(target: String): Map[String, Seq[String]] = lightModel.annotate(target)

  def annotate(target: Array[String]): Array[Map[String, Seq[String]]] =
    lightModel.annotate(target)

  def annotateJava(target: String): java.util.Map[String, java.util.List[String]] =
    lightModel.annotateJava(target)

  def annotateJava(targets: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[String]]] = {
    lightModel.annotateJava(targets)
  }

  def fullAnnotate(target: String, optionalTarget: String = ""): Map[String, Seq[IAnnotation]] = {
    lightModel.fullAnnotate(target, optionalTarget)
  }

  def fullAnnotate(targets: Array[String]): Array[Map[String, Seq[IAnnotation]]] = {
    lightModel.fullAnnotate(targets)
  }

  def fullAnnotate(
      targets: Array[String],
      optionalTargets: Array[String]): Array[Map[String, Seq[IAnnotation]]] = {
    lightModel.fullAnnotate(targets, optionalTargets)
  }

  def fullAnnotate(audio: Array[Float]): Map[String, Seq[IAnnotation]] = {
    lightModel.fullAnnotate(audio)
  }

  def fullAnnotate(audios: Array[Array[Float]]): Array[Map[String, Seq[IAnnotation]]] = {
    lightModel.fullAnnotate(audios)
  }

  def fullAnnotateJava(target: String): java.util.Map[String, java.util.List[IAnnotation]] = {
    lightModel.fullAnnotateJava(target)
  }

  def fullAnnotateJava(
      target: String,
      optionalTarget: String): java.util.Map[String, java.util.List[IAnnotation]] = {
    lightModel.fullAnnotateJava(target, optionalTarget)
  }

  def fullAnnotateJava(targets: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {
    lightModel.fullAnnotateJava(targets)
  }

  def fullAnnotateJava(
      targets: java.util.ArrayList[String],
      optionalTargets: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {
    lightModel.fullAnnotateJava(targets, optionalTargets)
  }

  def fullAnnotateSingleAudioJava(
      audio: java.util.ArrayList[Double]): java.util.Map[String, java.util.List[IAnnotation]] = {
    lightModel.fullAnnotateSingleAudioJava(audio)
  }

  def fullAnnotateAudiosJava(audios: java.util.ArrayList[java.util.ArrayList[Double]])
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {
    lightModel.fullAnnotateAudiosJava(audios)
  }

  def transform(dataFrame: DataFrame): DataFrame = model.transform(dataFrame)

}

object PretrainedPipeline {
  def fromDisk(path: String, parseEmbeddings: Boolean = false): PretrainedPipeline = {
    PretrainedPipeline(null, null, null, parseEmbeddings, Some(path))
  }
  def fromDisk(path: String): PretrainedPipeline = {
    fromDisk(path, false)
  }
}
