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

package com.johnsnowlabs.ml.ai

import ai.onnxruntime.OnnxTensor
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.Preprocessor

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

private[johnsnowlabs] class Wav2Vec2(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    configProtoBytes: Option[Array[Byte]] = None,
    vocabs: Map[String, BigInt],
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  val _tfWav2Vec2Signatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  private val wordDelimiterToken = "|"
  private val padVocabId = vocabs.getOrElse("<pad>", 0)
  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions
  private def sessionWarmup(): Unit = {
    val bufferedSource =
      scala.io.Source.fromInputStream(getClass.getResourceAsStream("/audio/audio_floats.csv"))

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close
    tag(Array(rawFloats, rawFloats), vocabs.toSeq.length)
  }

  sessionWarmup()

  def tag(batch: Array[Array[Float]], vocabSize: Int): Array[Int] = {

    val rawScores =
      detectedEngine match {
        case TensorFlow.name =>
          val tensors = new TensorResources()

          val audioTensors = tensors.createTensor(batch)

          val runner = tensorflowWrapper.get
            .getTFSessionWithSignature(configProtoBytes = configProtoBytes, initAllTables = false)
            .runner

          runner
            .feed(
              _tfWav2Vec2Signatures
                .getOrElse(ModelSignatureConstants.AudioValuesInput.key, "missing_input_values"),
              audioTensors)
            .fetch(_tfWav2Vec2Signatures
              .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

          val outs = runner.run().asScala

          tensors.clearTensors()
          audioTensors.close()
          val output = TensorResources.extractFloats(outs.head)
          tensors.clearSession(outs)
          output

        case ONNX.name =>
          val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)
          val audioTensors =
            OnnxTensor.createTensor(env, batch)
          val inputs =
            Map("input_values" -> audioTensors).asJava
          try {
            val results = runner.run(inputs)
            try {
              results
                .get("logits")
                .get()
                .asInstanceOf[OnnxTensor]
                .getFloatBuffer
                .array()
            } finally if (results != null) results.close()
          } catch {
            case e: Exception =>
              // Handle exceptions by logging or other means.
              e.printStackTrace()
              Array.empty[Float] // Return an empty array or appropriate error handling
          } finally {
            // Close tensors outside the try-catch to avoid repeated null checks.
            // These resources are initialized before the try-catch, so they should be closed here.
            audioTensors.close()
          }
      }
    rawScores
      .grouped(vocabSize)
      .toArray
      .map(x => x.indexOf(x.max))
  }

  def predict(
      audios: Array[AnnotationAudio],
      batchSize: Int,
      preprocessor: Preprocessor): Seq[Annotation] = {

    audios
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = encode(batch, preprocessor)
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

  // TODO: implement different padding strategies: max_length, longest, truncate
  def encode(
      annotations: Array[AnnotationAudio],
      preprocessor: Preprocessor): Array[Array[Float]] = {

    val maxLengthInBatch = annotations.map(x => x.result.length).max

    annotations.map { annot =>
      val normalized = if (preprocessor.do_normalize) normalize(annot.result) else annot.result
      val padding = Array.fill(maxLengthInBatch - normalized.length)(preprocessor.padding_value)

      preprocessor.padding_side match {
        case "left" => padding ++ normalized
        case "right" => normalized ++ padding
        case _ => normalized ++ padding
      }

    }

  }

  def normalize(rawAudio: Array[Float]): Array[Float] = {
    val normalizedData = ArrayBuffer[Float]()
    val meanData = mean(rawAudio)
    val varianceData = variance(rawAudio)
    for (x <- rawAudio) {
      normalizedData += (x - meanData) / scala.math.sqrt(varianceData + 0.0000001).toFloat
    }
    normalizedData.toArray
  }

  def mean(rawAudio: Array[Float]): Float =
    if (rawAudio.isEmpty) 0 else rawAudio.sum / rawAudio.length

  def variance(rawAudio: Array[Float]): Double = {
    val avg = mean(rawAudio)
    rawAudio.map(a => math.pow(a - avg, 2)).sum / rawAudio.length
  }

  // TODO: Check for infinite recursion
  def removeDup(lst: List[Int]): List[Int] = {
    lst match {
      case head :: tail => {
        val (duplicateList, remainList) = lst.span(_ == head)
        duplicateList.head :: removeDup(remainList)
      }
      case Nil => List()
    }
  }

  def decode(vocabs: Map[String, BigInt], vocabIds: Array[Int], batchSize: Int): Array[String] = {
    val noPadIdVocab = vocabIds.filter(tokenId => tokenId != padVocabId)
    val uniqueVocabIds = removeDup(noPadIdVocab.toList)
    uniqueVocabIds.grouped(uniqueVocabIds.length / batchSize).toArray.map { tokenIds =>
      tokenIds
        .map(tokenId => vocabs.find(vocab => vocab._2 == tokenId).map(_._1).getOrElse(""))
        .map(tokenId => if (tokenId == wordDelimiterToken) " " else tokenId)
        .mkString("")
    }
  }

}
