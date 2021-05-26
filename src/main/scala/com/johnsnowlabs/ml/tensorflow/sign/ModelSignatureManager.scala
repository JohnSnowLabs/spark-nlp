/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.tensorflow.sign

import org.slf4j.{Logger, LoggerFactory}
import org.tensorflow.SavedModelBundle
import org.tensorflow.proto.framework.TensorInfo
import org.tensorflow.proto.util.SaverDef

import java.util
import scala.util.matching.Regex


object ModelSignatureManager {

  val KnownProviders: Array[String] = Array("TF1", "TF2")

  private[ModelSignatureManager] val logger: Logger = LoggerFactory.getLogger("ModelSignatureManager")

  def apply(tfSignatureType: String = "TF1",
            tokenIdsValue: String = ModelSignatureConstants.InputIdsV1.value,
            maskIdsValue: String = ModelSignatureConstants.AttentionMaskV1.value,
            segmentIdsValue: String = ModelSignatureConstants.TokenTypeIdsV1.value,
            embeddingsValue: String = ModelSignatureConstants.LastHiddenStateV1.value,
            sentenceEmbeddingsValue: String = ModelSignatureConstants.PoolerOutputV1.value): Map[String, String] =

    tfSignatureType.toUpperCase match {
      case "TF1" =>
        Map[String, String](
          ModelSignatureConstants.InputIds.key -> tokenIdsValue,
          ModelSignatureConstants.AttentionMask.key -> maskIdsValue,
          ModelSignatureConstants.TokenTypeIds.key -> segmentIdsValue,
          ModelSignatureConstants.LastHiddenState.key -> embeddingsValue,
          ModelSignatureConstants.PoolerOutput.key -> sentenceEmbeddingsValue)
      case _ => throw new Exception("Model provider not available.")
    }

  def getInputIdsKey: String = ModelSignatureConstants.InputIds.key

  def getInputIdsValue: String = ModelSignatureConstants.InputIds.value

  def getAttentionMaskIdsKey: String = ModelSignatureConstants.AttentionMask.key

  def getAttentionMaskIdsValue: String = ModelSignatureConstants.AttentionMask.value

  def getTokenTypeIdsKey: String = ModelSignatureConstants.TokenTypeIds.key

  def getTokenTypeIdsValue: String = ModelSignatureConstants.TokenTypeIds.value

  def getLastHiddenStateKey: String = ModelSignatureConstants.LastHiddenState.key

  def getLastHiddenStateValue: String = ModelSignatureConstants.LastHiddenState.value

  def getPoolerOutputKey: String = ModelSignatureConstants.PoolerOutput.key

  def getPoolerOutputValue: String = ModelSignatureConstants.PoolerOutput.value

  /** Return a formatted map of key -> value for model signature objects */
  def convertToAdoptedKeys(matched: Map[String, String]): Map[String, String] = {
    val SecondaryIndexSep = "::"
    matched
      .map { case (k, v) => k.split(SecondaryIndexSep)(1) -> v } // signature def name
      .map { case (k, v) => ModelSignatureConstants.toAdoptedKeys(k) -> v }
  }

  /** Extract signatures from actual model
   *
   * @param model : a SavedModelBundle object
   * @return a list of tuples of type (OperationType, key, TFInfoName)
   * */
  def getSignaturesFromModel(model: SavedModelBundle): Map[String, String] = {
    import collection.JavaConverters._

    val InputPrefix = "input"
    val OutputPrefix = "output"
    val Sep = "::"

    val modelSignatures = scala.collection.mutable.Map.empty[String, String]

    /**
     * Loop imperatively over signature definition to extract them in a map
     *
     * @param prefix             : input or output attribute
     * @param signDefinitionsMap : Java signature definition map
     * */

    def extractSignatureDefinitions(prefix: String, signDefinitionsMap: util.Map[String, TensorInfo]): Unit = {
      for (e <- signDefinitionsMap.entrySet.asScala) {

        val key: String = e.getKey
        val tfInfo: TensorInfo = e.getValue

        modelSignatures +=
          (s"$prefix$Sep$key$Sep${ModelSignatureConstants.Name.key}" ->
            tfInfo.getName)
        modelSignatures +=
          (s"$prefix$Sep$key$Sep${ModelSignatureConstants.DType.key}" ->
            tfInfo.getDtype.toString)
        modelSignatures +=
          (s"$prefix$Sep$key$Sep${ModelSignatureConstants.DimCount.key}" ->
            tfInfo.getTensorShape.getDimCount.toString)
        modelSignatures +=
          (s"$prefix$Sep$key$Sep${ModelSignatureConstants.ShapeDimList.key}" ->
            tfInfo.getTensorShape.getDimList.toString.replaceAll("\n", "").replaceAll("size:", ""))
        modelSignatures +=
          (s"$prefix$Sep$key$Sep${ModelSignatureConstants.SerializedSize.key}" ->
            tfInfo.getName)
      }
    }

    if (model.metaGraphDef.hasGraphDef && model.metaGraphDef.getSignatureDefCount > 0) {
      for (sigDef <- model.metaGraphDef.getSignatureDefMap.values.asScala) {
        // extract input sign map
        extractSignatureDefinitions(InputPrefix, sigDef.getInputsMap)
        // extract output sign map
        extractSignatureDefinitions(OutputPrefix, sigDef.getOutputsMap)
      }
    }

    modelSignatures.toMap
  }

  /** Regex matcher */
  def findTFKeyMatch(candidate: String, pattern: Regex): Boolean = {
    val _value = candidate.split("::")(1) // i.e. input::input_ids::name
    val res = pattern findAllIn _value
    if (res.nonEmpty)
      true
    else
      false
  }

  /**
   * Extract the model provider counting the signature pattern matches
   *
   * @param signDefNames  : the candidate signature definitions inputs and outputs
   * @param modelProvider : the true model provider in between TF1 and TF2 to evaluate
   * @return : the model provider name in between TF1 and TF2
   * */
  def classifyProvider(signDefNames: Map[String, String], modelProvider: Option[String] = None): String = {

    val versionMatchesCount = KnownProviders.map { provider =>
      provider -> {
        signDefNames.map { signName =>
          val patterns: Array[Regex] = ModelSignatureConstants.getSignaturePatterns(provider)
          val matches = (for (pattern <- patterns if findTFKeyMatch(signName._1, pattern)) yield 1).toList.sum
          matches
        }
      }.sum
    }.toMap

    val (topModelProvider, _) = versionMatchesCount.toSeq.maxBy(_._2)
    topModelProvider
  }

  /**
   * Extract input and output signatures from TF saved models
   *
   * @param modelProvider model framework provider, i.e. TF1 or TF2, default TF1
   * @param model         loaded SavedModelBundle
   * @return the list ot matching signatures as tuples
   * */
  def extractSignatures(model: SavedModelBundle, saverDef: SaverDef): Option[Map[String, String]] = {

    val signatureCandidates = getSignaturesFromModel(model)
    val signDefNames: Map[String, String] = signatureCandidates.filterKeys(_.contains(ModelSignatureConstants.Name.key))

    val modelProvider = classifyProvider(signDefNames)

    val adoptedKeys = convertToAdoptedKeys(signDefNames) + (
      "filenameTensorName_" -> saverDef.getFilenameTensorName.replaceAll(":0", ""),
      "restoreOpName_" -> saverDef.getRestoreOpName.replaceAll(":0", ""),
      "saveTensorName_" -> saverDef.getSaveTensorName.replaceAll(":0", "")
    )

    Option(adoptedKeys)
  }
}
