/*
 * Copyright 2017-2021 John Snow Labs
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
import com.johnsnowlabs.ml.tensorflow.wrap.TFWrapper
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.tensorflow.ndarray.buffer.IntDataBuffer

import scala.collection.JavaConverters._

/**
 *
 * @param tensorflowWrapper    Bert Model wrapper with TensorFlow Wrapper
 * @param sentenceStartTokenId Id of sentence start Token
 * @param sentenceEndTokenId   Id of sentence end Token.
 * @param configProtoBytes     Configuration for TensorFlow session
 * @param tags                 labels which model was trained with in order
 * @param signatures           TF v2 signatures in Spark NLP
 * */
class TensorflowRoBertaClassification(val tensorflowWrapper: TFWrapper[_],
                                      sentenceStartTokenId: Int,
                                      sentenceEndTokenId: Int,
                                      padTokenId: Int,
                                      configProtoBytes: Option[Array[Byte]] = None,
                                      tags: Map[String, Int],
                                      signatures: Option[Map[String, String]] = None
                                     ) extends Serializable {

  val _tfRoBertaSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())

  /** Encode the input sequence to indexes IDs adding padding where necessary */
  def encode(sentences: Seq[(WordpieceTokenizedSentence, Int)], maxSequenceLength: Int): Seq[Array[Int]] = {
    val maxSentenceLength =
      Array(
        maxSequenceLength - 2,
        sentences.map { case (wpTokSentence, _) => wpTokSentence.tokens.length }.max).min

    sentences
      .map { case (wpTokSentence, _) =>
        val tokenPieceIds = wpTokSentence.tokens.map(t => t.pieceId)
        val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(padTokenId)

        Array(sentenceStartTokenId) ++ tokenPieceIds.take(maxSentenceLength) ++ Array(sentenceEndTokenId) ++ padding
      }
  }

  def calcluateSoftmax(scores: Array[Float]): Array[Float] = {
    val exp = scores.map(x => math.exp(x))
    exp.map(x => x / exp.sum).map(_.toFloat)
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    val tensors = new TensorResources()

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val batchLength = batch.length

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers.offset(offset).write(sentence.map(x => if (x == padTokenId) 0 else 1))
      }

    val runner = tensorflowWrapper.getTFHubSession(configProtoBytes = configProtoBytes, initAllTables = false).runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)

    runner
      .feed(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"), tokenTensors)
      .feed(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"), maskTensors)
      .fetch(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    val dim = rawScores.length / (batchLength * maxSentenceLength)
    val batchScores: Array[Array[Array[Float]]] = rawScores.grouped(dim).map(scores =>
      calcluateSoftmax(scores)).toArray.grouped(maxSentenceLength).toArray

    batchScores
  }

  def predict(sentences: Seq[WordpieceTokenizedSentence],
              originalTokenSentences: Seq[TokenizedSentence],
              batchSize: Int,
              maxSentenceLength: Int
             ): Seq[Annotation] = {

    /*Run calculation by batches*/
    sentences.zipWithIndex.grouped(batchSize).flatMap { batch =>
      val encoded = encode(batch, maxSentenceLength)
      val logits = tag(encoded)

      /*Combine tokens and calculated logits*/
      batch.zip(logits).flatMap { case (sentence, tokenVectors) =>
        val tokenLength = sentence._1.tokens.length

        /*All wordpiece logits*/
        val tokenLogits = tokenVectors.slice(1, tokenLength + 1)

        /*Word-level and span-level alignment with Tokenizer
        https://github.com/google-research/bert#tokenization

        ### Input
        orig_tokens = ["John", "Johanson", "'s",  "house"]
        labels      = ["NNP",  "NNP",      "POS", "NN"]

        # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
        # orig_to_tok_map == [1, 2, 4, 6]*/

        val labelsWithScores = sentence._1.tokens.zip(tokenLogits).flatMap {
          case (token, scores) =>
            originalTokenSentences(sentence._2).indexedTokens.find(
              p => p.begin == token.begin).map {
              token =>
                val label = tags.find(_._2 == scores.zipWithIndex.maxBy(_._1)._2).map(_._1).getOrElse("NA")
                val meta = scores.zipWithIndex.flatMap(x => Map(tags.find(_._2 == x._2).map(_._1).toString -> x._1.toString))
                Annotation(
                  annotatorType = AnnotatorType.NAMED_ENTITY,
                  begin = token.begin,
                  end = token.end,
                  result = label,
                  metadata = Map("sentence" -> sentence._2.toString) ++ meta
                )
            }
        }
        labelsWithScores.toSeq
      }
    }.toSeq
  }

}


