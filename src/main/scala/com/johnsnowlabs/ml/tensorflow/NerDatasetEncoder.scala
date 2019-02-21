package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common.WordpieceEmbeddingsSentence

class NerDatasetEncoder
(
  val params: DatasetEncoderParams
) {

  private val nonDefaultTags = params.tags
    .filter(_ != params.defaultTag)
    .zipWithIndex
    .map(p => (p._1, p._2 + 1))
    .toMap

  val tag2Id = Map(params.defaultTag -> 0) ++ nonDefaultTags
  val tags = tag2Id
    .map(p => (p._2, p._1))
    .toArray
    .sortBy(p => p._1)
    .map(p => p._2)

  val chars = params.chars.toArray

  val char2Id = params.chars.zip(1 to params.chars.length).toMap

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
    val wordLengths = sentences.map{
      sentence =>
        val lengths = sentence.tokens.map(word => word.wordpiece.length)
        Range(0, maxSentenceLength)
          .map{idx => getOrElse(lengths, idx, 0)}
          .toArray
    }

    if (wordLengths.flatten.isEmpty) {
      return NerBatch.empty
    }

    val maxWordLength = wordLengths.flatten.max

    val wordEmbeddings =
    Range(0, batchSize).map{i =>
      val sentence = sentences(i)
      Range(0, maxSentenceLength).map{j =>
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

  def encodeTags(tags: Array[Array[String]]): Array[Array[Int]] = {
    val batchSize = tags.length
    val maxSentence = tags.map(t => t.length).max

    (0 until batchSize).map{i =>
      (0 until maxSentence).map{j =>
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
    * @param tags 2-dimensional tensor in plain array
    * @param sentenceLength Every sentence length (number of words).
    * @return List of tags for each sentence
    */
  def convertBatchTags(tags: Array[String], sentenceLength: Array[Int]): Array[Array[String]] = {
    val sentences = sentenceLength.length
    val maxSentenceLength = tags.length / sentences

    Range(0, sentences).map{i =>
      Range(0, sentenceLength(i)).map{j =>
        tags(i * maxSentenceLength + j)
      }.toArray
    }.toArray
  }
}

/**
  * Batch that contains data in Tensorflow input format.
  */
class NerBatch (
  // Word vector representation. Shape: Batch x Max Sentence Length x Embeddings Dim
  val wordEmbeddings: Array[Array[Array[Float]]],

  // Char ids for every word in every sentence. Shape: Batch x Max Sentence Length x Max Word length
  val charIds: Array[Array[Array[Int]]],

  // Word Length of every sentence. Shape: Batch x Max Sentence Length
  val wordLengths: Array[Array[Int]],

  // Length of every batch sentence. Shape: Batch
  val sentenceLengths: Array[Int],

  // Max length of sentence
  val maxLength: Int,

  // Is current wordpiece is token start? Shape: Batch x Max Sentence Length
  val isWordStart: Array[Array[Boolean]]
)
{
  def batchSize: Int = wordEmbeddings.length
}

object NerBatch {
  def empty = new NerBatch(Array.empty, Array.empty, Array.empty, Array.empty, 0, Array.empty)
}


case class DatasetEncoderParams
(
  tags: List[String],
  chars: List[Char],
  emptyVector: List[Float],
  embeddingsDim: Int,
  defaultTag: String = "O"
) {
  val emptyEmbeddings = emptyVector.toArray
}
