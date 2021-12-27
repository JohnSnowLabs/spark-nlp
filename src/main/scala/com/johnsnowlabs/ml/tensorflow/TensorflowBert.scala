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
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.embeddings.TransformerEmbeddings
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.slf4j.{Logger, LoggerFactory}
import org.tensorflow.ndarray.buffer.IntDataBuffer

import scala.collection.JavaConverters._

/**
 * BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture
 *
 *
 * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddingsTestSpec.scala]] for further reference on how to use this API.
 * Sources:
 *
 * @param tensorflowWrapper    Bert Model wrapper with TensorFlow Wrapper
 * @param sentenceStartTokenId Id of sentence start Token
 * @param sentenceEndTokenId   Id of sentence end Token.
 * @param configProtoBytes     Configuration for TensorFlow session
 *
 *                             Paper:  [[ https://arxiv.org/abs/1810.04805]]
 *
 *                             Source:  [[https://github.com/google-research/bert]]
 * */
class TensorflowBert(val tensorflowWrapper: TensorflowWrapper,
                     val sentenceStartTokenId: Int,
                     val sentenceEndTokenId: Int,
                     configProtoBytes: Option[Array[Byte]] = None,
                     signatures: Option[Map[String, String]] = None,
                     vocabulary: Map[String, Int]
                    ) extends Serializable with TransformerEmbeddings {

  val _tfBertSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())

  override protected val sentencePadTokenId: Int = 0

  override def tokenizeWithAlignment(tokens: Seq[TokenizedSentence], caseSensitive: Boolean,
                                     maxSentenceLength: Int): Seq[WordpieceTokenizedSentence] = {
    val basicTokenizer = new BasicTokenizer(caseSensitive)
    val encoder = new WordpieceEncoder(vocabulary)

    tokens.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens = tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map { token =>
        val content = if (caseSensitive) token.token else token.token.toLowerCase()
        val sentenceBegin = token.begin
        val sentenceEnd = token.end
        val sentenceInedx = tokenIndex.sentenceIndex
        val result = basicTokenizer.tokenize(Sentence(content, sentenceBegin, sentenceEnd, sentenceInedx))
        if (result.nonEmpty) result.head else IndexedToken("")
      }
      val wordpieceTokens = bertTokens.flatMap(token => encoder.encode(token)).take(maxSentenceLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  override def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    val tensors = new TensorResources()

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val batchLength = batch.length

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val segmentBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers.offset(offset).write(sentence.map(x => if (x == 0) 0 else 1))
        segmentBuffers.offset(offset).write(Array.fill(maxSentenceLength)(0))
      }

    val runner = tensorflowWrapper.getTFSessionWithSignature(configProtoBytes = configProtoBytes, savedSignatures = signatures, initAllTables = false).runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)
    val segmentTensors = tensors.createIntBufferTensor(shape, segmentBuffers)

    runner
      .feed(_tfBertSignatures.getOrElse(ModelSignatureConstants.InputIdsV1.key, "missing_input_id_key"), tokenTensors)
      .feed(_tfBertSignatures.getOrElse(ModelSignatureConstants.AttentionMaskV1.key, "missing_input_mask_key"), maskTensors)
      .feed(_tfBertSignatures.getOrElse(ModelSignatureConstants.TokenTypeIdsV1.key, "missing_segment_ids_key"), segmentTensors)
      .fetch(_tfBertSignatures.getOrElse(ModelSignatureConstants.LastHiddenStateV1.key, "missing_sequence_output_key"))

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()

    val dim = embeddings.length / (batchLength * maxSentenceLength)
    val shrinkedEmbeddings: Array[Array[Array[Float]]] =
      embeddings
        .grouped(dim).toArray
        .grouped(maxSentenceLength).toArray

    val emptyVector = Array.fill(dim)(0f)

    batch.zip(shrinkedEmbeddings).map { case (ids, embeddings) =>
      if (ids.length > embeddings.length) {
        embeddings.take(embeddings.length - 1) ++
          Array.fill(embeddings.length - ids.length)(emptyVector) ++
          Array(embeddings.last)
      } else {
        embeddings
      }
    }

  }

  def predictSequence(tokens: Seq[WordpieceTokenizedSentence],
                      sentences: Seq[Sentence],
                      batchSize: Int,
                      maxSentenceLength: Int,
                      isLong: Boolean = false
                     ): Seq[Annotation] = {

    /*Run embeddings calculation by batches*/
    tokens.zip(sentences).zipWithIndex.grouped(batchSize).flatMap { batch =>
      val tokensBatch = batch.map(x => (x._1._1, x._2))
      val sentencesBatch = batch.map(x => x._1._2)
      val encoded = encode(tokensBatch, maxSentenceLength)
      val embeddings = if (isLong) {
        tagSequenceSBert(encoded)
      } else {
        tagSequence(encoded)
      }

      sentencesBatch.zip(embeddings).map { case (sentence, vectors) =>
        val metadata = Map("sentence" -> sentence.index.toString,
          "token" -> sentence.content,
          "pieceId" -> "-1",
          "isWordStart" -> "true"
        )
        val finalMetadata = if (sentence.metadata.isDefined) {
          sentence.metadata.getOrElse(Map.empty) ++ metadata
        } else {
          metadata
        }
        Annotation(
          annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
          begin = sentence.start,
          end = sentence.end,
          result = sentence.content,
          metadata = finalMetadata,
          embeddings = vectors
        )
      }
    }.toSeq
  }

  def tagSequence(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    val tensors = new TensorResources()
    val tensorsMasks = new TensorResources()
    val tensorsSegments = new TensorResources()

    val maxSentenceLength = batch.map(x => x.length).max
    val batchLength = batch.length

    val tokenBuffers = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers = tensorsMasks.createIntBuffer(batchLength * maxSentenceLength)
    val segmentBuffers = tensorsSegments.createIntBuffer(batchLength * maxSentenceLength)


    val shape = Array(batchLength.toLong, maxSentenceLength)

    batch.zipWithIndex.foreach { case (sentence, idx) =>
      val offset = idx * maxSentenceLength

      tokenBuffers.offset(offset).write(sentence)
      maskBuffers.offset(offset).write(sentence.map(x => if (x == 0) 0 else 1))
      segmentBuffers.offset(offset).write(Array.fill(maxSentenceLength)(0))
    }

    val runner = tensorflowWrapper.getTFSessionWithSignature(configProtoBytes = configProtoBytes, savedSignatures = signatures, initAllTables = false).runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensorsMasks.createIntBufferTensor(shape, maskBuffers)
    val segmentTensors = tensorsSegments.createIntBufferTensor(shape, segmentBuffers)

    runner
      .feed(_tfBertSignatures.getOrElse(ModelSignatureConstants.InputIdsV1.key, "missing_input_id_key"), tokenTensors)
      .feed(_tfBertSignatures.getOrElse(ModelSignatureConstants.AttentionMaskV1.key, "missing_input_mask_key"), maskTensors)
      .feed(_tfBertSignatures.getOrElse(ModelSignatureConstants.TokenTypeIdsV1.key, "missing_segment_ids_key"), segmentTensors)
      .fetch(_tfBertSignatures.getOrElse(ModelSignatureConstants.PoolerOutput.key, "missing_pooled_output_key"))

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()

    val dim = embeddings.length / batchLength
    embeddings.grouped(dim).toArray

  }

  def tagSequenceSBert(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    val tensors = new TensorResources()

    val maxSentenceLength = batch.map(x => x.length).max
    val batchLength = batch.length

    val tokenBuffers = tensors.createLongBuffer(batchLength * maxSentenceLength)
    val maskBuffers = tensors.createLongBuffer(batchLength * maxSentenceLength)
    val segmentBuffers = tensors.createLongBuffer(batchLength * maxSentenceLength)

    val shape = Array(batchLength.toLong, maxSentenceLength)

    batch.zipWithIndex.foreach { case (sentence, idx) =>
      val offset = idx * maxSentenceLength
      tokenBuffers.offset(offset).write(sentence.map(_.toLong))
      maskBuffers.offset(offset).write(sentence.map(x => if (x == 0L) 0L else 1L))
      segmentBuffers.offset(offset).write(Array.fill(maxSentenceLength)(0L))
    }


    val runner = tensorflowWrapper.getTFSessionWithSignature(configProtoBytes = configProtoBytes, savedSignatures = signatures, initAllTables = false).runner

    val tokenTensors = tensors.createLongBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createLongBufferTensor(shape, maskBuffers)
    val segmentTensors = tensors.createLongBufferTensor(shape, segmentBuffers)

    runner
      .feed(_tfBertSignatures.getOrElse(ModelSignatureConstants.InputIdsV1.key, "missing_input_id_key"), tokenTensors)
      .feed(_tfBertSignatures.getOrElse(ModelSignatureConstants.AttentionMaskV1.key, "missing_input_mask_key"), maskTensors)
      .feed(_tfBertSignatures.getOrElse(ModelSignatureConstants.TokenTypeIdsV1.key, "missing_segment_ids_key"), segmentTensors)
      .fetch(_tfBertSignatures.getOrElse(ModelSignatureConstants.PoolerOutput.key, "missing_pooled_output_key"))

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()

    val dim = embeddings.length / batchLength
    embeddings.grouped(dim).toArray
  }

}

object TensorflowBert {
  private[TensorflowBert] val logger: Logger = LoggerFactory.getLogger("TensorflowBert")
}
