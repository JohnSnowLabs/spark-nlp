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

package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common.WordpieceEmbeddingsSentence

import scala.collection.Map

class NerDatasetEncoder(val params: DatasetEncoderParams) extends Serializable {

  private val nonDefaultTags = params.tags
    .filter(_ != params.defaultTag)
    .zipWithIndex
    .map(p => (p._1, p._2 + 1))
    .toMap

  val tag2Id: Map[String, Int] = Map(params.defaultTag -> 0) ++ nonDefaultTags
  val tags: Array[String] = tag2Id
    .map(p => (p._2, p._1))
    .toArray
    .sortBy(p => p._1)
    .map(p => p._2)

  val chars: Array[Char] = params.chars.toArray

  val char2Id: Map[Char, Int] = params.chars.zip(1 to params.chars.length).toMap

  def getOrElse[T](source: Array[T], i: Int, value: => T): T = {
    if (i < source.length)
      source(i)
    else
      value
  }

  def encodeInputData(sentences: Array[WordpieceEmbeddingsSentence]): NerBatch = {

    val batchSize = sentences.length

    if (batchSize == 0)
      return NerBatch.empty

    val sentenceLengths = sentences.map(s => s.tokens.length)
    val maxSentenceLength = sentenceLengths.max

    if (maxSentenceLength == 0)
      return NerBatch.empty

    val wordLengths = sentences.map {
      sentence =>
        val lengths = sentence.tokens.map(word => word.wordpiece.length)
        Range(0, maxSentenceLength)
          .map { idx => getOrElse(lengths, idx, 0) }
          .toArray
    }


    assert(wordLengths.flatten.nonEmpty, "")
    if (wordLengths.flatten.isEmpty) {
      return NerBatch.empty
    }

    val maxWordLength = wordLengths.flatten.max

    val wordEmbeddings =
      Range(0, batchSize).map { i =>
        val sentence = sentences(i)
        Range(0, maxSentenceLength).map { j =>
          if (j < sentence.tokens.length)
            sentence.tokens(j).embeddings
          else
            params.emptyEmbeddings

        }.toArray
      }.toArray

    val charIds =
      Range(0, batchSize).map { i =>
        val sentence = sentences(i)
        Range(0, maxSentenceLength).map { j =>
          val word = (if (j < sentence.tokens.length)
            sentence.tokens(j).wordpiece
          else
            "").toCharArray

          Range(0, maxWordLength).map { k =>
            val char = getOrElse(word, k, Char.MinValue)
            char2Id.getOrElse(char, 0)
          }.toArray
        }.toArray
      }.toArray

    val isWordStart = sentences.map { sentence =>
      Range(0, maxSentenceLength).map { j =>
        if (j < sentence.tokens.length)
          sentence.tokens(j).isWordStart
        else
          false
      }.toArray
    }

    new NerBatch(
      wordEmbeddings,
      charIds,
      wordLengths,
      sentenceLengths,
      maxSentenceLength,
      isWordStart
    )
  }

  /**
   * Converts Tag names to Identifiers
   *
   * @param tags batches of labels/classes for each sentence/document
   * @return batches of tag ids for each sentence/document
   */
  def encodeTags(tags: Array[Array[String]]): Array[Array[Int]] = {
    val batchSize = tags.length
    val maxSentence = tags.map(t => t.length).max

    (0 until batchSize).map { i =>
      (0 until maxSentence).map { j =>
        val tag = getOrElse(tags(i), j, params.defaultTag)
        tag2Id.getOrElse(tag, 0)
      }.toArray
    }.toArray
  }

  /**
   * Converts Tag Identifiers to Source Names
   *
   * @param tagIds Tag Ids encoded for Tensorflow Model.
   * @return Tag names
   */
  def decodeOutputData(tagIds: Array[Int]): Array[String] = {
    tagIds.map(id => getOrElse(tags, id, params.defaultTag))
  }

  /**
   * Converts Tensorflow tags output to 2-dimensional Array with shape: (Batch, Sentence Length).
   *
   * @param predictedTags 2-dimensional tensor in plain array
   * @param allTags All original tags
   * @param sentenceLength Every sentence length (number of words).
   * @return List of tags for each sentence
   */
  def convertBatchTags(predictedTags: Array[String],
                       allTags: Array[String],
                       sentenceLength: Array[Int],
                       prob: Option[Seq[Array[Float]]]): Array[Array[(String, Option[Array[Map[String, String]]])]] = {

    val sentences = sentenceLength.length
    val maxSentenceLength = predictedTags.length / sentences

    Range(0, sentences).map { i =>
      Range(0, sentenceLength(i)).map { j => {
        val index = i * maxSentenceLength + j
        val metaWithProb: Option[Array[Map[String, String]]] = {
          if(prob.isDefined)
            Some(allTags
              .zipWithIndex
              .map {case (t, i) => Map(t -> prob.map(_ (index)).getOrElse(Array.empty[String]).lift(i).getOrElse(0.0f).toString)})
          else None
        }

        (predictedTags(index), metaWithProb)
      }
      }.toArray
    }.toArray
  }
}

/**
 * Batch that contains data in Tensorflow input format.
 *
 * @param wordEmbeddings  Word vector representation. Shape: Batch x Max Sentence Length x Embeddings Dim
 * @param charIds         Char ids for every word in every sentence. Shape: Batch x Max Sentence Length x Max Word length
 * @param wordLengths     Word Length of every sentence. Shape: Batch x Max Sentence Length
 * @param sentenceLengths Length of every batch sentence. Shape: Batch
 * @param maxLength       Max length of sentence
 * @param isWordStart     Is current wordpiece is token start? Shape: Batch x Max Sentence Length
 */
class NerBatch(
                val wordEmbeddings: Array[Array[Array[Float]]],
                val charIds: Array[Array[Array[Int]]],
                val wordLengths: Array[Array[Int]],
                val sentenceLengths: Array[Int],
                val maxLength: Int,
                val isWordStart: Array[Array[Boolean]]
              ) {
  def batchSize: Int = wordEmbeddings.length
}

object NerBatch {
  def empty = new NerBatch(Array.empty, Array.empty, Array.empty, Array.empty, 0, Array.empty)
}

/**
 *
 * @param tags list of unique tags
 * @param chars list of unique characters
 * @param emptyVector list of embeddings
 * @param embeddingsDim dimension of embeddings
 * @param defaultTag the default tag
 */
case class DatasetEncoderParams
(
  tags: List[String],
  chars: List[Char],
  emptyVector: List[Float],
  embeddingsDim: Int,
  defaultTag: String = "O"
) {
  val emptyEmbeddings: Array[Float] = emptyVector.toArray
}
