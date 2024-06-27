/*
 * Copyright 2017-2024 John Snow Labs
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

package com.johnsnowlabs.nlp.gguf

import com.johnsnowlabs.ml.gguf.GGUFWrapper
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import de.kherud.llama.LlamaModel
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods

/** Annotator that uses the llama.cpp library to generate text completions.
  *
  * The annotator requires a GGUF model, which needs to be provided either by either providing a
  * path to a local file or a URL (TODO).
  *
  * For settable parameters, and their explanations, see [[HasLlamaCppProperties]] and refer to
  * the llama.cpp documentation of
  * [[https://github.com/ggerganov/llama.cpp/tree/7d5e8777ae1d21af99d4f95be10db4870720da91/examples/server server.cpp]]
  * for more information.
  *
  * ==Example==
  *
  * TODO
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class AutoGGUFModel(override val uid: String)
    extends AnnotatorModel[AutoGGUFModel]
    with HasBatchedAnnotate[AutoGGUFModel]
    with HasEngine
    with HasLlamaCppProperties
    with HasProtectedParams {

  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DOCUMENT
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.DOCUMENT)

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("AutoGGUFModel"))

  private var _model: Option[Broadcast[GGUFWrapper]] = None

  /** @group getParam */
  def getModelIfNotSet: GGUFWrapper = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, wrapper: GGUFWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(spark.sparkContext.broadcast(wrapper))
    }
    this
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getModelIfNotSet.saveToFile(path)
  }

  /** Completes the batch of annotations.
    *
    * @param batchedAnnotations
    *   Annotations (single element arrays) in batches
    * @return
    *   Completed text sequences
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    val annotations: Seq[Annotation] = batchedAnnotations.flatten
    if (annotations.nonEmpty) {

      val modelParams =
        getModelParameters.setNParallel(getBatchSize) // set parallel decoding to batch size
      val inferenceParams = getInferenceParameters

      println("DEBUG DHA: modelParams: " + modelParams.toString)
      println("DEBUG DHA: inferenceParams " + inferenceParams.toString)

      val model: LlamaModel = getModelIfNotSet.getSession(modelParams)

      val annotationsText = annotations.map(_.result)
      val completed_texts = model.requestBatchCompletion(annotationsText.toArray, inferenceParams)

      val result: Seq[Seq[Annotation]] =
        annotations.zip(completed_texts).map { case (anno, text) =>
          Seq(
            new Annotation(
              outputAnnotatorType,
              0, // TODO Maybe prepend the original text?
              text.length - 1,
              text,
              Map("prompt" -> anno.result)))
        }
      result
    } else Seq(Seq.empty[Annotation])
  }

  def getMetadataMap: Map[String, String] = {
    val metadataJsonString = getMetadata
    if (metadataJsonString.isEmpty) Map.empty
    else {
      implicit val formats: DefaultFormats.type = DefaultFormats
      JsonMethods.parse(metadataJsonString).extract[Map[String, String]]
    }
  }
}

trait ReadablePretrainedAutoGGUFModelModel
    extends ParamsAndFeaturesReadable[AutoGGUFModel]
    with HasPretrained[AutoGGUFModel] {
  override val defaultModelName: Some[String] = Some("TODO") // TODO: might not even be needed?
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): AutoGGUFModel = super.pretrained()

  override def pretrained(name: String): AutoGGUFModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): AutoGGUFModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): AutoGGUFModel =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadAutoGGUFModelDLModel {
  this: ParamsAndFeaturesReadable[AutoGGUFModel] =>

  val suffix: String = "TODO"

  def readModel(instance: AutoGGUFModel, path: String, spark: SparkSession): Unit = {
    def findGGUFModelInFolder(): String = {
      val folder = new java.io.File(path)
      if (folder.exists && folder.isDirectory) {
        folder.listFiles
          .filter(_.isFile)
          .filter(_.getName.endsWith(".gguf"))
          .map(_.getAbsolutePath)
          .headOption // Should only be one file
          .getOrElse(throw new IllegalArgumentException(s"Could not find GGUF model in $path"))
      } else {
        throw new IllegalArgumentException(s"Path $path is not a directory")
      }
    }

    val model = AutoGGUFModel.loadSavedModel(findGGUFModelInFolder(), spark)
    instance.setModelIfNotSet(spark, model.getModelIfNotSet)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): AutoGGUFModel = {
    // TODO copyToLocal and potentially enable download from HF-URLS
    val localPath: String = ResourceHelper.copyToLocal(modelPath)
    val annotatorModel = new AutoGGUFModel()
    annotatorModel
      .setModelIfNotSet(spark, GGUFWrapper.read(spark, localPath))

    val metadata = LlamaModel.getMetadataFromFile(localPath)
    if (metadata.nonEmpty) annotatorModel.setMetadata(metadata)
    annotatorModel
  }
}

/** This is the companion object of [[AutoGGUFModel]]. Please refer to that class for the
  * documentation.
  */
object AutoGGUFModel extends ReadablePretrainedAutoGGUFModelModel with ReadAutoGGUFModelDLModel
