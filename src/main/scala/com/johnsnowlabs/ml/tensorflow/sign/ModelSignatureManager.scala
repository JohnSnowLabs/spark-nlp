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
import org.tensorflow.proto.framework.{SignatureDef, TensorInfo}

import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex
import java.util


object ModelSignatureManager {

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
      .map{case(k, v) => k.split(SecondaryIndexSep)(1) -> v}// signature def name
      .map{case(k, v) => ModelSignatureConstants.toAdoptedKeys(k) -> v}
  }

  /** Extract signatures from actual model
   *
   * @param model : a SavedModelBundle object
   * @return a list of tuples of type (OperationType, key, TFInfoName)
   * */
  def getSignaturesFromModel(model: SavedModelBundle) = {
    import collection.JavaConverters._

    val InputPrefix = "input"
    val OutputPrefix = "output"
    val Sep = "::"

    val modelSignatures = scala.collection.mutable.Map.empty[String, String]

    def extractSignDefInputs(sigDef: SignatureDef) = {
      val inputs: util.Map[String, TensorInfo] = sigDef.getInputsMap
      for (e <- inputs.entrySet.asScala) {
        val key: String = e.getKey
        val tfInfo: TensorInfo = e.getValue

        modelSignatures += (s"$InputPrefix$Sep$key$Sep${ModelSignatureConstants.Name.key}" -> tfInfo.getName)
        modelSignatures += (s"$InputPrefix$Sep$key$Sep${ModelSignatureConstants.DType.key}" -> tfInfo.getDtype.toString)
        modelSignatures += (s"$InputPrefix$Sep$key$Sep${ModelSignatureConstants.DimCount.key}" -> tfInfo.getTensorShape.getDimCount.toString)
        modelSignatures += (s"$InputPrefix$Sep$key$Sep${ModelSignatureConstants.ShapeDimList.key}" -> tfInfo.getTensorShape.getDimList.toString.replaceAll("\n", "").replaceAll("size:", ""))
        modelSignatures += (s"$InputPrefix$Sep$key$Sep${ModelSignatureConstants.SerializedSize.key}" -> tfInfo.getName)
      }
    }

    def extractSignDefOutputs(sigDef: SignatureDef) = {
      val outputs: util.Map[String, TensorInfo] = sigDef.getOutputsMap
      for (e <- outputs.entrySet.asScala) {
        val key: String = e.getKey
        val tfInfo: TensorInfo = e.getValue

        modelSignatures += (s"$OutputPrefix$Sep$key$Sep${ModelSignatureConstants.Name.key}" -> tfInfo.getName)
        modelSignatures += (s"$OutputPrefix$Sep$key$Sep${ModelSignatureConstants.DType.key}" -> tfInfo.getDtype.toString)
        modelSignatures += (s"$OutputPrefix$Sep$key$Sep${ModelSignatureConstants.DimCount.key}" -> tfInfo.getTensorShape.getDimCount.toString)
        modelSignatures += (s"$OutputPrefix$Sep$key$Sep${ModelSignatureConstants.ShapeDimList.key}" -> tfInfo.getTensorShape.getDimList.toString.replaceAll("\n", "").replaceAll("size:", ""))
        modelSignatures += (s"$OutputPrefix$Sep$key$Sep${ModelSignatureConstants.SerializedSize.key}" -> tfInfo.getName)
      }
    }

    if (model.metaGraphDef.hasGraphDef && model.metaGraphDef.getSignatureDefCount > 0) {
      for (sigDef <- model.metaGraphDef.getSignatureDefMap.values.asScala) {
        extractSignDefInputs(sigDef)
        extractSignDefOutputs(sigDef)
      }
    }

    modelSignatures.toMap
  }

  /**
   * Extract input and output signatures from TF saved models
   *
   * @param modelProvider model framework provider, i.e. TF1 or TF2, default TF1
   * @param model loaded SavedModelBundle
   * @return the list ot matching signatures as tuples
   * */
  def extractSignatures(modelProvider: String = "TF1", model: SavedModelBundle): Option[Map[String, String]] = {

    val signatureCandidates = getSignaturesFromModel(model)

    /** Regex matcher */
    def findTFKeyMatch(candidate: String, key: Regex) = {
      val pattern = key
      val res = pattern.unapplySeq(candidate)
      res.isDefined
    }

    /**
     * Extract matches from candidate key and model signatures
     *
     * @param candidate     : the candidate key name
     * @param modelProvider : the model provider in between default, TF2 and HF to select the proper keys
     * @return a list of matching keys as strings
     * */
    def extractCandidateMatches(candidate: String, modelProvider: String): List[String] = {
      val ReferenceKeys: Array[Regex] = ModelSignatureConstants.getSignaturePatterns(modelProvider)

      val matches = (
        for (refKey <- ReferenceKeys if findTFKeyMatch(candidate, refKey)) yield {
          refKey
        }).toList
      if (matches.isEmpty) List("N/A") else matches.mkString(",").split(",").toList
    }

    val signDefNames = signatureCandidates.filterKeys(_.contains(ModelSignatureConstants.Name.key))
    Option(convertToAdoptedKeys(signDefNames))
  }
}
