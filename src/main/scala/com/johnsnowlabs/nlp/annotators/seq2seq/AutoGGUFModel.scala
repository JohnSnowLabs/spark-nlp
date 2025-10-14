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
package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.gguf.GGUFWrapper
import com.johnsnowlabs.ml.gguf.GGUFWrapper.findGGUFModelInFolder
import com.johnsnowlabs.ml.util.LlamaCPP
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.llama.LlamaExtensions
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import de.kherud.llama.{InferenceParameters, LlamaException, LlamaModel}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** Annotator that uses the llama.cpp library to generate text completions with large language
  * models.
  *
  * For settable parameters, and their explanations, see [[HasLlamaCppInferenceProperties]],
  * [[HasLlamaCppModelProperties]] and refer to the llama.cpp documentation of
  * [[https://github.com/ggerganov/llama.cpp/tree/7d5e8777ae1d21af99d4f95be10db4870720da91/examples/server server.cpp]]
  * for more information.
  *
  * If the parameters are not set, the annotator will default to use the parameters provided by
  * the model.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val autoGGUFModel = AutoGGUFModel.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("completions")
  * }}}
  * The default model is `"Phi_4_mini_instruct_Q4_K_M_gguf"`, if no name is provided.
  *
  * AutoGGUFModel is also able to load pretrained models from [[AutoGGUFVisionModel]]. Just
  * specify the same name for the `pretrained` method, and it will load the text-part of the
  * multimodal model automatically.
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFModelTest.scala AutoGGUFModelTest]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFModel.ipynb example notebook]].
  *
  * ==Note==
  * To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
  * the number of GPU layers with the `setNGpuLayers` method.
  *
  * When using larger models, we recommend adjusting GPU usage with `setNCtx` and `setNGpuLayers`
  * according to your hardware to avoid out-of-memory errors.
  *
  * ==Example==
  *
  * {{{
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  * import spark.implicits._
  *
  * val document = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val autoGGUFModel = AutoGGUFModel
  *   .pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("completions")
  *   .setBatchSize(4)
  *   .setNPredict(20)
  *   .setNGpuLayers(99)
  *   .setTemperature(0.4f)
  *   .setTopK(40)
  *   .setTopP(0.9f)
  *   .setPenalizeNl(true)
  *
  * val pipeline = new Pipeline().setStages(Array(document, autoGGUFModel))
  *
  * val data = Seq("Hello, I am a").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  * result.select("completions").show(truncate = false)
  * +-----------------------------------------------------------------------------------------------------------------------------------+
  * |completions                                                                                                                        |
  * +-----------------------------------------------------------------------------------------------------------------------------------+
  * |[{document, 0, 78,  new user.  I am currently working on a project and I need to create a list of , {prompt -> Hello, I am a}, []}]|
  * +-----------------------------------------------------------------------------------------------------------------------------------+
  * }}}
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
    with HasLlamaCppModelProperties
    with HasLlamaCppInferenceProperties
    with HasProtectedParams
    with CompletionPostProcessing {

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

  /** Closes the llama.cpp model backend freeing resources. The model is reloaded when used again.
    */
  def close(): Unit = GGUFWrapper.closeBroadcastModel(_model)

  private[johnsnowlabs] def setEngine(engineName: String): this.type = set(engine, engineName)

  setDefault(
    engine -> LlamaCPP.name,
    useChatTemplate -> true,
    nCtx -> 4096,
    nBatch -> 512,
    nPredict -> 100,
    nGpuLayers -> 99,
    systemPrompt -> "You are a helpful assistant.",
    batchSize -> 2)

  /** Sets the number of parallel processes for decoding. This is an alias for `setBatchSize`.
    *
    * @group setParam
    * @param nParallel
    *   The number of parallel processes for decoding
    */
  def setNParallel(nParallel: Int): this.type = {
    setBatchSize(nParallel)
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
    // TODO: group by doc and sentence
    if (annotations.nonEmpty) {
      val annotationsText = annotations.map(_.result).toArray

      val modelParams =
        getModelParameters.setParallel(getBatchSize) // set parallel decoding to batch size
      val inferenceParams: InferenceParameters = getInferenceParameters

      val model: LlamaModel = getModelIfNotSet.getSession(modelParams)

      val (completedTexts: Array[String], metadata: Map[String, String]) =
        try {
          val results: Array[String] = LlamaExtensions.multiComplete(
            model,
            inferenceParams,
            getSystemPrompt,
            annotationsText)
          val resultsCleaned = processCompletions(results)
          (resultsCleaned, Map.empty)
        } catch {
          case e: LlamaException =>
            logger.error("Error in llama.cpp batch completion", e)
            (Array.fill(annotationsText.length)(""), Map("llamacpp_exception" -> e.getMessage))
        }
      annotations.zip(completedTexts).map { case (annotation, text) =>
        Seq(
          new Annotation(
            outputAnnotatorType,
            0,
            text.length - 1,
            text,
            annotation.metadata ++ metadata))
      }
    } else Seq(Seq.empty[Annotation])
  }
}

trait ReadablePretrainedAutoGGUFModel
    extends ParamsAndFeaturesFallbackReadable[AutoGGUFModel]
    with HasPretrained[AutoGGUFModel] {
  override val defaultModelName: Some[String] = Some("Phi_4_mini_instruct_Q4_K_M_gguf")
  override val defaultLang: String = "en"

  /** Java compliant-overrides */
  override def pretrained(): AutoGGUFModel = super.pretrained()

  override def pretrained(name: String): AutoGGUFModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): AutoGGUFModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): AutoGGUFModel =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadAutoGGUFModel {
  this: ParamsAndFeaturesFallbackReadable[AutoGGUFModel] =>

  override def fallbackLoad(folder: String, spark: SparkSession): AutoGGUFModel = {
    val actualFolderPath: String = ResourceHelper.resolvePath(folder)
    val localFolder = ResourceHelper.copyToLocal(actualFolderPath)
    val modelFile = findGGUFModelInFolder(localFolder)
    loadSavedModel(modelFile, spark)
  }

  def readModel(instance: AutoGGUFModel, path: String, spark: SparkSession): Unit = {
    val model: GGUFWrapper = GGUFWrapper.readModel(path, spark)
    instance.setModelIfNotSet(spark, model)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): AutoGGUFModel = {
    // TODO potentially enable download from HF-URLS
    val localPath: String = ResourceHelper.copyToLocal(modelPath)
    val annotatorModel = new AutoGGUFModel()
    annotatorModel
      .setModelIfNotSet(spark, GGUFWrapper.read(spark, localPath))
      .setEngine(LlamaCPP.name)

    val metadata = LlamaExtensions.getMetadataFromFile(localPath)
    if (metadata.nonEmpty) annotatorModel.setMetadata(metadata)
    annotatorModel
  }
}

/** This is the companion object of [[AutoGGUFModel]]. Please refer to that class for the
  * documentation.
  */
object AutoGGUFModel extends ReadablePretrainedAutoGGUFModel with ReadAutoGGUFModel
