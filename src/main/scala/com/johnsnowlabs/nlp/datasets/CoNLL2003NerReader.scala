package com.johnsnowlabs.nlp.datasets

import java.io.File

import com.johnsnowlabs.ml.crf.{CrfDataset, DatasetMetadata, InstanceLabels, TextSentenceLabels}
import com.johnsnowlabs.nlp.AnnotatorType
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.common.{TaggedSentence, TokenPieceEmbeddings, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.annotators.ner.crf.{DictionaryFeatures, FeatureGenerator}
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddingsFormat, WordEmbeddingsIndexer, WordEmbeddingsRetriever}
import com.johnsnowlabs.nlp.util.io.ExternalResource

/**
  * Helper class for to work with CoNLL 2003 dataset for NER task
  * Class is made for easy use from Java
  */
class CoNLL2003NerReader(wordEmbeddingsFile: String,
                         wordEmbeddingsNDims: Int,
                         normalize: Boolean,
                         embeddingsFormat: WordEmbeddingsFormat.Format,
                         possibleExternalDictionary: Option[ExternalResource]) {

  private val nerReader = CoNLL()

  private var wordEmbeddings: WordEmbeddingsRetriever = _

  if (wordEmbeddingsFile != null) {
    require(new File(wordEmbeddingsFile).exists())

    var fileDb = wordEmbeddingsFile + ".db"

    if (!new File(fileDb).exists()) {
      embeddingsFormat match {
        case WordEmbeddingsFormat.TEXT =>
          WordEmbeddingsIndexer.indexText(wordEmbeddingsFile, fileDb)
        case WordEmbeddingsFormat.BINARY =>
          WordEmbeddingsIndexer.indexBinary(wordEmbeddingsFile, fileDb)
        case WordEmbeddingsFormat.SPARKNLP =>
          fileDb = wordEmbeddingsFile
      }
    }

    if (new File(fileDb).exists()) {
      wordEmbeddings = WordEmbeddingsRetriever(fileDb, wordEmbeddingsNDims, normalize)
    }
  }

  private val fg = FeatureGenerator(
    DictionaryFeatures.read(possibleExternalDictionary)
  )

  private def resolveEmbeddings(sentences: Seq[PosTaggedSentence]): Seq[WordpieceEmbeddingsSentence] = {
    sentences.zipWithIndex.map { case (s, idx) =>
      val tokens = s.indexedTaggedWords.map{token =>
        val vector = wordEmbeddings.getEmbeddingsVector(token.word)
        new TokenPieceEmbeddings(token.word, token.word,
          -1, true, vector,
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