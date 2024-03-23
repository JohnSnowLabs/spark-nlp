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
import de.kherud.llama.args._
import de.kherud.llama.{InferenceParameters, LlamaModel, ModelParameters}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** TODO
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

  override def onWrite(path: String, spark: SparkSession): Unit = ???

  /** Completes the batch of annotations.
    *
    * @param batchedAnnotations
    *   Audio annotations in batches
    * @return
    *   Completed text sequences
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    batchedAnnotations.map { annotations: Array[Annotation] =>
      println(s"Processing batch of length ${annotations.length}")
      println(s"First prompt: ${annotations.head.result}")

      val inferenceParams = new InferenceParameters("")
        .setInputPrefix(getInputPrefix)
        .setInputSuffix(getInputSuffix)
        .setCachePrompt(getCachePrompt)
        .setNPredict(getNPredict)
        .setTopK(getTopK)
        .setTopP(getTopP)
        .setMinP(getMinP)
        .setTfsZ(getTfsZ)
        .setTypicalP(getTypicalP)
        .setTemperature(getTemperature)
        .setDynamicTemperatureRange(getDynamicTemperatureRange)
        .setDynamicTemperatureExponent(getDynamicTemperatureExponent)
        .setRepeatLastN(getRepeatLastN)
        .setRepeatPenalty(getRepeatPenalty)
        .setFrequencyPenalty(getFrequencyPenalty)
        .setPresencePenalty(getPresencePenalty)
        .setMiroStat(MiroStat.values.apply(getMiroStat))
        .setMiroStatTau(getMiroStatTau)
        .setMiroStatEta(getMiroStatEta)
        .setPenalizeNl(getPenalizeNl)
        .setNKeep(getNKeep)
        .setSeed(getSeed)
        .setNProbs(getNProbs)
        .setMinKeep(getMinKeep)
        .setGrammar(getGrammar)
        .setPenaltyPrompt(getPenaltyPrompt)
        .setIgnoreEos(getIgnoreEos)
        .setStopStrings(getStopStrings: _*)
        .setUseChatTemplate(getUseChatTemplate)

      val modelParams = new ModelParameters()
        .setNThreads(getNThreads)
        .setNThreadsDraft(getNThreadsDraft)
        .setNThreadsBatch(getNThreadsBatch)
        .setNThreadsBatchDraft(getNThreadsBatchDraft)
        .setNCtx(getNCtx)
        .setNBatch(getNBatch)
        .setNUbatch(getNUbatch)
        .setNDraft(getNDraft)
        .setNChunks(getNChunks)
        .setNParallel(getNParallel)
        .setNSequences(getNSequences)
        .setPSplit(getPSplit)
        .setNGpuLayers(getNGpuLayers)
        .setNGpuLayersDraft(getNGpuLayersDraft)
        .setSplitMode(GpuSplitMode.values()(getSplitMode))
        .setMainGpu(getMainGpu)
        .setTensorSplit(getTensorSplit.map(_.toFloat))
        .setNBeams(getNBeams)
        .setGrpAttnN(getGrpAttnN)
        .setGrpAttnW(getGrpAttnW)
        .setRopeFreqBase(getRopeFreqBase)
        .setRopeFreqScale(getRopeFreqScale)
        .setYarnExtFactor(getYarnExtFactor)
        .setYarnAttnFactor(getYarnAttnFactor)
        .setYarnBetaFast(getYarnBetaFast)
        .setYarnBetaSlow(getYarnBetaSlow)
        .setYarnOrigCtx(getYarnOrigCtx)
        .setDefragmentationThreshold(getDefragmentationThreshold)
        .setNuma(NumaStrategy.values()(getNuma))
        .setRopeScalingType(RopeScalingType.values()(getRopeScalingType))
        .setPoolingType(PoolingType.values()(getPoolingType))
        .setModelDraft(getModelDraft)
        .setLookupCacheStaticFilePath(getLookupCacheStaticFilePath)
        .setLookupCacheDynamicFilePath(getLookupCacheDynamicFilePath)
        .setLoraBase(getLoraBase)
        .setEmbedding(getEmbedding)
        .setContinuousBatching(getContinuousBatching)
        .setFlashAttention(getFlashAttention)
        .setInputPrefixBos(getInputPrefixBos)
        .setUseMmap(getUseMmap)
        .setUseMlock(getUseMlock)
        .setNoKvOffload(getNoKvOffload)
        .setSystemPrompt(getSystemPrompt)
        .setChatTemplate(getChatTemplate)

      val model: LlamaModel = getModelIfNotSet.getSession(modelParams)

      if (annotations.nonEmpty) {
        val annotationsText = annotations.map(_.result)
        val completed_texts = model.requestBatchCompletion(annotationsText, inferenceParams)

        val result: Seq[Annotation] = annotations.zip(completed_texts).map { case (anno, text) =>
          new Annotation(
            outputAnnotatorType,
            0, // TODO Maybe prepend the original text?
            text.length - 1,
            text,
            Map("prompt" -> anno.result))
        }
        result
      } else Seq.empty[Annotation]
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

  def readModel(instance: AutoGGUFModel, path: String, spark: SparkSession): Unit = ???
  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): AutoGGUFModel = {
    // TODO copyToLocal and potentially enable download from HF-URLS
    // val localPath: String = ResourceHelper.copyToLocal(path)
    // TODO: extract parameters
    val annotatorModel = new AutoGGUFModel()

    annotatorModel.setModelIfNotSet(spark, GGUFWrapper.read(spark, modelPath))
    annotatorModel
  }
}

/** This is the companion object of [[AutoGGUFModel]]. Please refer to that class for the
  * documentation.
  */
object AutoGGUFModel extends ReadablePretrainedAutoGGUFModelModel with ReadAutoGGUFModelDLModel
