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
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.tensorflow.ndarray.buffer.IntDataBuffer

import scala.collection.JavaConverters._

/** The DistilBERT model was proposed in the paper '''DistilBERT, a distilled version of BERT:
  * smaller, faster, cheaper and lighter''' [[https://arxiv.org/abs/1910.01108]]. DistilBERT is a
  * small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40%
  * less parameters than `bert-base-uncased`, runs 60% faster while preserving over 95% of BERT's
  * performances as measured on the GLUE language understanding benchmark.
  *
  * The abstract from the paper is the following:
  *
  * As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural
  * Language Processing (NLP), operating these large models in on-the-edge and/or under
  * constrained computational training or inference budgets remains challenging. In this work, we
  * propose a method to pre-train a smaller general-purpose language representation model, called
  * DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like
  * its larger counterparts. While most prior work investigated the use of distillation for
  * building task-specific models, we leverage knowledge distillation during the pretraining phase
  * and show that it is possible to reduce the size of a BERT model by 40%, while retaining 97% of
  * its language understanding capabilities and being 60% faster. To leverage the inductive biases
  * learned by larger models during pretraining, we introduce a triple loss combining language
  * modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is
  * cheaper to pre-train and we demonstrate its capabilities for on-device computations in a
  * proof-of-concept experiment and a comparative on-device study.
  *
  * Tips:
  *
  *   - DistilBERT doesn't have :obj:`token_type_ids`, you don't need to indicate which token
  *     belongs to which segment. Just separate your segments with the separation token
  *     :obj:`tokenizer.sep_token` (or :obj:`[SEP]`).
  *
  *   - DistilBERT doesn't have options to select the input positions (:obj:`position_ids` input).
  *     This could be added if necessary though, just let us know if you need this option.
  *
  * @param tensorflowWrapper
  *   Bert Model wrapper with TensorFlow Wrapper
  * @param sentenceStartTokenId
  *   Id of sentence start Token
  * @param sentenceEndTokenId
  *   Id of sentence end Token.
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  */
class TensorflowDistilBert(
    val tensorflowWrapper: TensorflowWrapper,
    sentenceStartTokenId: Int,
    sentenceEndTokenId: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  val _tfBertSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())

  /** Encode the input sequence to indexes IDs adding padding where necessary */
  def encode(
      sentences: Seq[(WordpieceTokenizedSentence, Int)],
      maxSequenceLength: Int): Seq[Array[Int]] = {
    val maxSentenceLength =
      Array(
        maxSequenceLength - 2,
        sentences.map { case (wpTokSentence, _) =>
          wpTokSentence.tokens.length
        }.max).min

    sentences
      .map { case (wpTokSentence, _) =>
        val tokenPieceIds = wpTokSentence.tokens.map(t => t.pieceId)
        val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(0)

        Array(sentenceStartTokenId) ++ tokenPieceIds.take(maxSentenceLength) ++ Array(
          sentenceEndTokenId) ++ padding
      }
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
        maskBuffers.offset(offset).write(sentence.map(x => if (x == 0) 0 else 1))
      }

    val runner = tensorflowWrapper
      .getTFSessionWithSignature(
        configProtoBytes = configProtoBytes,
        savedSignatures = signatures,
        initAllTables = false)
      .runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)

    runner
      .feed(
        _tfBertSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfBertSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .fetch(_tfBertSignatures
        .getOrElse(ModelSignatureConstants.LastHiddenState.key, "missing_sequence_output_key"))

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    val dim = embeddings.length / (batchLength * maxSentenceLength)
    val shrinkedEmbeddings: Array[Array[Array[Float]]] =
      embeddings.grouped(dim).toArray.grouped(maxSentenceLength).toArray

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

  /** @param batch
    *   batches of sentences
    * @return
    *   batches of vectors for each sentence
    */
  def tagSequence(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    val tensors = new TensorResources()
    val tensorsMasks = new TensorResources()

    val maxSentenceLength = batch.map(x => x.length).max
    val batchLength = batch.length

    val tokenBuffers = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers = tensorsMasks.createIntBuffer(batchLength * maxSentenceLength)

    val shape = Array(batchLength.toLong, maxSentenceLength)

    batch.zipWithIndex.foreach { case (sentence, idx) =>
      val offset = idx * maxSentenceLength

      tokenBuffers.offset(offset).write(sentence)
      maskBuffers.offset(offset).write(sentence.map(x => if (x == 0) 0 else 1))
    }

    val runner = tensorflowWrapper
      .getTFSessionWithSignature(configProtoBytes = configProtoBytes, initAllTables = false)
      .runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensorsMasks.createIntBufferTensor(shape, maskBuffers)

    runner
      .feed(
        _tfBertSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfBertSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .fetch(_tfBertSignatures
        .getOrElse(ModelSignatureConstants.PoolerOutput.key, "missing_pooled_output_key"))

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()

    val dim = embeddings.length / batchLength
    embeddings.grouped(dim).toArray

  }

  def predict(
      sentences: Seq[WordpieceTokenizedSentence],
      originalTokenSentences: Seq[TokenizedSentence],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean): Seq[WordpieceEmbeddingsSentence] = {

    /*Run embeddings calculation by batches*/
    sentences.zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = encode(batch, maxSentenceLength)
        val vectors = tag(encoded)

        /*Combine tokens and calculated embeddings*/
        batch.zip(vectors).map { case (sentence, tokenVectors) =>
          val tokenLength = sentence._1.tokens.length

          /*All wordpiece embeddings*/
          val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)
          val originalIndexedTokens = originalTokenSentences(sentence._2)

          /*Word-level and span-level alignment with Tokenizer
        https://github.com/google-research/bert#tokenization

        ### Input
        orig_tokens = ["John", "Johanson", "'s",  "house"]
        labels      = ["NNP",  "NNP",      "POS", "NN"]

        # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
        # orig_to_tok_map == [1, 2, 4, 6]*/

          val tokensWithEmbeddings =
            sentence._1.tokens.zip(tokenEmbeddings).flatMap { case (token, tokenEmbedding) =>
              val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
              val originalTokensWithEmbeddings = originalIndexedTokens.indexedTokens
                .find(p => p.begin == tokenWithEmbeddings.begin)
                .map { token =>
                  val originalTokenWithEmbedding = TokenPieceEmbeddings(
                    TokenPiece(
                      wordpiece = tokenWithEmbeddings.wordpiece,
                      token = if (caseSensitive) token.token else token.token.toLowerCase(),
                      pieceId = tokenWithEmbeddings.pieceId,
                      isWordStart = tokenWithEmbeddings.isWordStart,
                      begin = token.begin,
                      end = token.end),
                    tokenEmbedding)
                  originalTokenWithEmbedding
                }
              originalTokensWithEmbeddings
            }

          WordpieceEmbeddingsSentence(tokensWithEmbeddings, originalIndexedTokens.sentenceIndex)
        }
      }
      .toSeq
  }

  def predictSequence(
      tokens: Seq[WordpieceTokenizedSentence],
      sentences: Seq[Sentence],
      batchSize: Int,
      maxSentenceLength: Int): Seq[Annotation] = {

    /*Run embeddings calculation by batches*/
    tokens
      .zip(sentences)
      .zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val tokensBatch = batch.map(x => (x._1._1, x._2))
        val sentencesBatch = batch.map(x => x._1._2)
        val encoded = encode(tokensBatch, maxSentenceLength)
        val embeddings = tagSequence(encoded)

        sentencesBatch.zip(embeddings).map { case (sentence, vectors) =>
          Annotation(
            annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
            begin = sentence.start,
            end = sentence.end,
            result = sentence.content,
            metadata = Map(
              "sentence" -> sentence.index.toString,
              "token" -> sentence.content,
              "pieceId" -> "-1",
              "isWordStart" -> "true"),
            embeddings = vectors)
        }
      }
      .toSeq
  }

}
