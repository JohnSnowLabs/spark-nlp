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

import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex

import java.util


object ModelSignatureManager {

  private[ModelSignatureManager] val logger: Logger = LoggerFactory.getLogger("ModelSignatureManager")

  def apply(tfSignatureType: String = "JSL",
            tokenIdsValue: String = ModelSignatureConstants.TokenIds.value,
            maskIdsValue: String = ModelSignatureConstants.MaskIds.value,
            segmentIdsValue: String = ModelSignatureConstants.SegmentIds.value,
            embeddingsValue: String = ModelSignatureConstants.LastHiddenState.value,
            sentenceEmbeddingsValue: String = ModelSignatureConstants.PoolerOutput.value): Map[String, String] =

    tfSignatureType.toUpperCase match {
      case "JSL" =>
        Map[String, String](
          ModelSignatureConstants.TokenIds.key -> tokenIdsValue,
          ModelSignatureConstants.MaskIds.key -> maskIdsValue,
          ModelSignatureConstants.SegmentIds.key -> segmentIdsValue,
          ModelSignatureConstants.LastHiddenState.key -> embeddingsValue,
          ModelSignatureConstants.PoolerOutput.key -> sentenceEmbeddingsValue)
      case _ => throw new Exception("Model provider not available.")
    }

  def getInputIdsKey: String = ModelSignatureConstants.TokenIds.key

  def getInputIdsValue: String = ModelSignatureConstants.TokenIds.value

  def getAttentionMaskIdsKey: String = ModelSignatureConstants.MaskIds.key

  def getAttentionMaskIdsValue: String = ModelSignatureConstants.MaskIds.value

  def getTokenTypeIdsKey: String = ModelSignatureConstants.SegmentIds.key

  def getTokenTypeIdsValue: String = ModelSignatureConstants.SegmentIds.value

  def getLastHiddenStateKey: String = ModelSignatureConstants.LastHiddenState.key

  def getLastHiddenStateValue: String = ModelSignatureConstants.LastHiddenState.value

  def getPoolerOutputKey: String = ModelSignatureConstants.PoolerOutput.key

  def getPoolerOutputValue: String = ModelSignatureConstants.PoolerOutput.value

  /** Return a formatted map of key -> value for model signature objects */
  def convertToAdoptedKeys(matched: List[((String, String, String), List[String])]): Map[String, String] = {
    matched.map(e => ModelSignatureConstants.toAdoptedKeys(e._1._2) -> e._1._3).toMap
  }

  /** Extract signatures from actual model
   *
   * @param model : a SavedModelBundle object
   * @return a list of tuples of type (OperationType, key, TFInfoName)
   * */
  def getSignaturesFromModel(model: SavedModelBundle): List[(String, String, String)] = {
    import collection.JavaConverters._

    val InputOp = "feed"
    val OutputOp = "fetch"

    val modelSignatures = new ListBuffer[(String, String, String)]()

    if (model.metaGraphDef.hasGraphDef && model.metaGraphDef.getSignatureDefCount > 0) {
      for (sigDef <- model.metaGraphDef.getSignatureDefMap.values.asScala) {
        val inputs: util.Map[String, TensorInfo] = sigDef.getInputsMap
        for (e <- inputs.entrySet.asScala) {
          val key: String = e.getKey
          val tfInfo: TensorInfo = e.getValue
          logger.debug("\nSignatureDef InputMap key: " + key + " tfInfo: " + tfInfo.getName)
          modelSignatures += ((InputOp, key, tfInfo.getName))
        }
      }
      for (sigDef <- model.metaGraphDef.getSignatureDefMap.values.asScala) {
        val outputs: util.Map[String, TensorInfo] = sigDef.getOutputsMap
        for (e <- outputs.entrySet.asScala) {
          val key: String = e.getKey
          val tfInfo: TensorInfo = e.getValue
          logger.debug("\nSignatureDef OutputMap key: " + key + " tfInfo: " + tfInfo.getName)
          modelSignatures += ((OutputOp, key, tfInfo.getName))
        }
      }
    }
    modelSignatures.toList
  }

  /**
   * Extract input and output signatures from TF saved models
   *
   * @param modelProvider model framework provider, i.e. default, TF2 or HF
   * @param model loaded SavedModelBundle
   * @return the list ot matching signatures as tuples
   * */
  def extractSignatures(modelProvider: String = "JSL",
                        model: SavedModelBundle): Option[Map[String, String]] = {

    val signatureCandidates = getSignaturesFromModel(model)
    logger.debug(signatureCandidates.toString)

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

    val matched = signatureCandidates.map(s => (s, extractCandidateMatches(s._2, modelProvider)))

    Option(convertToAdoptedKeys(matched))
  }
}
