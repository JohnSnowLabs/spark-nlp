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

package com.johnsnowlabs.nlp.training

import java.io.File

import com.johnsnowlabs.ml.crf.{CrfDataset, DatasetMetadata, InstanceLabels, TextSentenceLabels}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.common.{TaggedSentence, TokenPieceEmbeddings, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.annotators.ner.crf.{DictionaryFeatures, FeatureGenerator}
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddingsBinaryIndexer, WordEmbeddingsReader, WordEmbeddingsTextIndexer, WordEmbeddingsWriter}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.storage.RocksDBConnection

/**
  * Helper class for to work with CoNLL 2003 dataset for NER task
  * Class is made for easy use from Java
  */
class CoNLL2003NerReader(wordEmbeddingsFile: String,
                         wordEmbeddingsNDims: Int,
                         normalize: Boolean,
                         embeddingsFormat: ReadAs.Value,
                         possibleExternalDictionary: Option[ExternalResource]) {

  private val nerReader = CoNLL(
    documentCol = "document",
    sentenceCol = "sentence",
    tokenCol = "token",
    posCol = "pos"
  )

  private var wordEmbeddings: WordEmbeddingsReader = _

  if (wordEmbeddingsFile != null) {
    require(new File(wordEmbeddingsFile).exists())

    var fileDb = wordEmbeddingsFile + ".db"
    val connection = RocksDBConnection.getOrCreate(fileDb)

    if (!new File(fileDb).exists()) {
      embeddingsFormat match {
        case ReadAs.TEXT =>
          WordEmbeddingsTextIndexer.index(wordEmbeddingsFile, new WordEmbeddingsWriter(connection, false, wordEmbeddingsNDims, 5000, 5000))
        case ReadAs.BINARY =>
          WordEmbeddingsBinaryIndexer.index(wordEmbeddingsFile, new WordEmbeddingsWriter(connection, false, wordEmbeddingsNDims, 5000, 5000))
      }
    }

    if (new File(fileDb).exists()) {
      wordEmbeddings = new WordEmbeddingsReader(connection, normalize, wordEmbeddingsNDims, 1000)
    }
  }

  private val fg = FeatureGenerator(
    DictionaryFeatures.read(possibleExternalDictionary)
  )

  private def resolveEmbeddings(sentences: Seq[PosTaggedSentence]): Seq[WordpieceEmbeddingsSentence] = {
    sentences.zipWithIndex.map { case (s, idx) =>
      val tokens = s.indexedTaggedWords.map{token =>
        val vectorOption = wordEmbeddings.lookup(token.word)
        TokenPieceEmbeddings(token.word, token.word,
          -1, true, vectorOption, Array.fill[Float](wordEmbeddingsNDims)(0f),
          token.begin, token.end)
      }

      WordpieceEmbeddingsSentence(tokens, idx)
    }
  }

  private def readDataset(er: ExternalResource)
  : Seq[(TextSentenceLabels, TaggedSentence, WordpieceEmbeddingsSentence)] = {

    val docs = nerReader.readDocs(er)
    val labels = docs.flatMap(_.nerTagged)
      .map(sentence => TextSentenceLabels(sentence.tags))

    val posTaggedSentences = docs.flatMap(_.posTagged)
    val withEmbeddings = resolveEmbeddings(posTaggedSentences)

    labels.zip(posTaggedSentences.zip(withEmbeddings))
      .map{case(l, (p, w)) => (l, p, w)}
  }

  def readNerDataset(er: ExternalResource, metadata: Option[DatasetMetadata] = None): CrfDataset = {
    val lines = readDataset(er)
    if (metadata.isEmpty)
      fg.generateDataset(lines)
    else {
      val labeledInstances = lines.map { line =>
        val instance = fg.generate(line._2, line._3, metadata.get)
        val labels = InstanceLabels(line._1.labels.map(l => metadata.get.label2Id.getOrElse(l, -1)))
        (labels, instance)
      }
      CrfDataset(labeledInstances, metadata.get)
    }
  }
}