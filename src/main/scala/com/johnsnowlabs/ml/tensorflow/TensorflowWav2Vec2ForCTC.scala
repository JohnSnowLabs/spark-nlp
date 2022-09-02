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

package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.Preprocessor

import scala.collection.JavaConverters._

class TensorflowWav2Vec2ForCTC(
    val tensorflowWrapper: TensorflowWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  val _tfWav2Vec2Signatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  def tag(
      batch: Array[Array[Float]],
      activation: String = ActivationFunction.softmax): Array[Array[Float]] = {
    val tensors = new TensorResources()
    val batchLength = batch.length

    val imageTensors = tensors.createTensor(batch)

    val runner = tensorflowWrapper
      .getTFSessionWithSignature(configProtoBytes = configProtoBytes, initAllTables = false)
      .runner

    runner
      .feed(
        _tfWav2Vec2Signatures
          .getOrElse(ModelSignatureConstants.AudioValuesInput.key, "missing_input_values"),
        imageTensors)
      .fetch(_tfWav2Vec2Signatures
        .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()
    imageTensors.close()

    val dim = rawScores.length / batchLength
    val batchScores: Array[Array[Float]] =
      rawScores
        .grouped(dim)
        .map(scores => calculateSoftmax(scores))
        .toArray
    batchScores
//    Array(Array.emptyFloatArray)
  }

  /** Calculate softmax from returned logits
    * @param scores
    *   logits output from output layer
    * @return
    */
  def calculateSoftmax(scores: Array[Float]): Array[Float] = {
    val exp = scores.map(x => math.exp(x))
    exp.map(x => x / exp.sum).map(_.toFloat)
  }

  /** Calculate sigmoid from returned logits
    * @param scores
    *   logits output from output layer
    * @return
    */
  def calculateSigmoid(scores: Array[Float]): Array[Float] = {
    scores.map(x => 1 / (1 + Math.exp(-x)).toFloat)
  }

  def predict(
      audios: Array[AnnotationAudio],
      batchSize: Int,
      vocabs: Map[String, BigInt],
      preprocessor: Preprocessor,
      activation: String = ActivationFunction.softmax): Seq[Annotation] = {

    audios
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = encode(batch, preprocessor)
        val logits = tag(encoded, activation)

        batch.zip(logits).map { case (image, score) =>
          val decodedSpeech = ""
          Annotation(
            annotatorType = AnnotatorType.DOCUMENT,
            begin = 0,
            end = decodedSpeech.length - 1,
            result = decodedSpeech,
            metadata = Map("audio" -> "0"))
        }

      }
  }.toSeq

  def encode(
      annotations: Array[AnnotationAudio],
      preprocessor: Preprocessor): Array[Array[Float]] = {
    annotations.map(x => x.result)

  }

  def encode(logits: Array[Array[Float]]) = ???

}
