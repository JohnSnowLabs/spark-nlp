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

  def tag(batch: Array[Array[Float]], vocabSize: Int): Array[Int] = {
    val tensors = new TensorResources()
    val batchLength = batch.length
    val maxSequenceLength = batch.map(x => x.length).max

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

    rawScores
      .grouped(vocabSize)
      .toArray
      .map(x => x.indexOf(x.max))
  }

  def predict(
      audios: Array[AnnotationAudio],
      batchSize: Int,
      vocabs: Map[String, BigInt],
      preprocessor: Preprocessor): Seq[Annotation] = {

    audios
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = batch.map(x => x.result)
        val vocabIds = tag(encoded, vocabs.toSeq.length)
        val decoded = decode(vocabs, vocabIds, encoded.length)

        batch.zip(decoded).map { case (annot, string) =>
          val decodedSpeech = string
          Annotation(
            annotatorType = AnnotatorType.DOCUMENT,
            begin = 0,
            end = string.length - 1,
            result = decodedSpeech,
            metadata = Map("audio" -> "0", "sentence" -> "0") ++ annot.metadata)
        }

      }
  }.toSeq

  def decode(vocabs: Map[String, BigInt], vocabIds: Array[Int], batchSize: Int): Array[String] = {
    // TODO: requires better space cleaning and removing repetitive tokens
    vocabIds.grouped(vocabIds.length / batchSize).toArray.map { y =>
      y.filter(x => x != 0)
        .map(x => vocabs.find(y => y._2 == x).map(_._1).getOrElse(""))
        .map(x => if (x == "|") " " else x)
        .mkString("")
    }
  }
}
