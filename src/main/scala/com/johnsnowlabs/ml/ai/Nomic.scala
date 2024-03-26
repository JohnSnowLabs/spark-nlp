/*
 * Copyright 2017 - 2023  John Snow Labs
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.johnsnowlabs.ml.ai

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.onnx.TensorResources.implicits._

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

private[johnsnowlabs] class Nomic(
    val onnxWrapper: Option[OnnxWrapper],
    sentenceStartTokenId: Int,
    sentenceEndTokenId: Int)
    extends Serializable {

  protected val logger: Logger = LoggerFactory.getLogger("NOMIC_EMBEDDINGS")

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  /** Get sentence embeddings for a batch of sentences
    * @param batch
    *   batch of sentences
    * @return
    *   sentence embeddings
    */
  private def getSentenceEmbedding(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val paddedBatch = batch.map(arr => padArrayWithZeros(arr, maxSentenceLength))
    val embeddings =
      getSentenceEmbeddingFromOnnx(paddedBatch, maxSentenceLength)
    embeddings
  }

  private def padArrayWithZeros(arr: Array[Int], maxLength: Int): Array[Int] = {
    if (arr.length >= maxLength) {
      arr
    } else {
      arr ++ Array.fill(maxLength - arr.length)(0)
    }
  }

  private def getSentenceEmbeddingFromOnnx(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Array[Float]] = {

    val inputIds = batch.map(x => x.map(x => x.toLong)).toArray
    val attentionMask = batch.map(sentence => sentence.map(x => if (x < 0L) 0L else 1L)).toArray

    val (session: OrtSession, env: OrtEnvironment) =
      onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors = OnnxTensor.createTensor(env, inputIds)
    val maskTensors = OnnxTensor.createTensor(env, attentionMask)
    val inputs: java.util.Map[String, OnnxTensor] =
      Map(
        OnnxSignatures.encoderInputIDs -> tokenTensors,
        OnnxSignatures.encoderAttentionMask -> maskTensors).asJava
    val encoderResults = session.run(inputs)

    val encoderStateBuffer =
      try {
        val encoderStateTensor = encoderResults
          .get(OnnxSignatures.encoderOutput)
          .get()
          .asInstanceOf[OnnxTensor]

        val shape = encoderStateTensor.getInfo.getShape
        encoderStateTensor.getFloatBuffer
          .array()
          .grouped(shape(1).toInt)
          .toArray
      } finally {
        if (encoderResults != null) encoderResults.close()
      }

    tokenTensors.close()
    maskTensors.close()

    encoderStateBuffer
  }

  /** Predict sentence embeddings for a batch of sentences
    * @param sentences
    *   sentences
    * @param tokenizedSentences
    *   tokenized sentences
    * @param batchSize
    *   batch size
    * @param maxSentenceLength
    *   max sentence length
    * @return
    */
  def predict(
      sentences: Seq[Annotation],
      tokenizedSentences: Seq[WordpieceTokenizedSentence],
      batchSize: Int,
      maxSentenceLength: Int): Seq[Annotation] = {

    tokenizedSentences
      .zip(sentences)
      .zipWithIndex
      .grouped(batchSize)
      .toArray
      .flatMap { batch =>
        val tokensBatch = batch.map(x => x._1._1.tokens)
        val tokens = tokensBatch.map(x =>
          Array(sentenceStartTokenId) ++ x
            .map(y => y.pieceId)
            .take(maxSentenceLength - 2) ++ Array(sentenceEndTokenId))

        val sentenceEmbeddings = getSentenceEmbedding(tokens)

        batch.zip(sentenceEmbeddings).map { case (sentence, vectors) =>
          Annotation(
            annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
            begin = sentence._1._2.begin,
            end = sentence._1._2.end,
            result = sentence._1._2.result,
            metadata = sentence._1._2.metadata,
            embeddings = vectors)
        }
      }
  }

  private object OnnxSignatures {
    val encoderInputIDs: String = "input_ids"
    val encoderAttentionMask: String = "attention_mask"

    val encoderOutput: String = "sentence_embedding"
  }
}
