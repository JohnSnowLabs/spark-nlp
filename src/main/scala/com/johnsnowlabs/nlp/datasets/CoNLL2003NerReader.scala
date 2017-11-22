package com.johnsnowlabs.nlp.datasets

import java.io.File

import com.johnsnowlabs.ml.crf.{CrfDataset, DatasetMetadata, InstanceLabels, TextSentenceLabels}
import com.johnsnowlabs.nlp.AnnotatorType
import com.johnsnowlabs.nlp.annotators.common.TaggedSentence
import com.johnsnowlabs.nlp.annotators.ner.crf.{DictionaryFeatures, FeatureGenerator}
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsFormat, WordEmbeddingsIndexer}

/**
  * Helper class for to work with CoNLL 2003 dataset for NER task
  * Class is made for easy use from Java
  */
class CoNLL2003NerReader(wordEmbeddingsFile: String,
                         wordEmbeddingsNDims: Int,
                         embeddingsFormat: WordEmbeddingsFormat.Format,
                         dictionaryFile: String) {

  private val nerReader = CoNLL(3, AnnotatorType.NAMED_ENTITY)
  private val posReader = CoNLL(1, AnnotatorType.POS)

  private var wordEmbeddings: Option[WordEmbeddings] = None

  if (wordEmbeddingsFile != null) {
    require(new File(wordEmbeddingsFile).exists())

    var fileDb = wordEmbeddingsFile + ".db"

    if (!new File(fileDb).exists()) {
      embeddingsFormat match {
        case WordEmbeddingsFormat.Text =>
          WordEmbeddingsIndexer.indexText(wordEmbeddingsFile, fileDb)
        case WordEmbeddingsFormat.Binary =>
          WordEmbeddingsIndexer.indexBinary(wordEmbeddingsFile, fileDb)
        case WordEmbeddingsFormat.SparkNlp =>
          fileDb = wordEmbeddingsFile
      }

    }

    if (new File(fileDb).exists()) {
      wordEmbeddings = Some(WordEmbeddings(fileDb, wordEmbeddingsNDims))
    }
  }

  private val dicts = if (dictionaryFile == null) Seq.empty[String] else Seq(dictionaryFile)

  private val fg = FeatureGenerator(
    DictionaryFeatures.read(dicts),
    wordEmbeddings
  )

  private def readDataset(file: String): Seq[(TextSentenceLabels, TaggedSentence)] = {
    val labels = nerReader.readDocs(file).flatMap(_._2)
      .map(sentence => TextSentenceLabels(sentence.tags))

    val posTaggedSentences = posReader.readDocs(file).flatMap(_._2)
    labels.zip(posTaggedSentences)
  }

  def readNerDataset(file: String, metadata: Option[DatasetMetadata] = None): CrfDataset = {
    val lines = readDataset(file)
    if (metadata.isEmpty)
      fg.generateDataset(lines)
    else {
      val labeledInstances = lines.map { line =>
        val instance = fg.generate(line._2, metadata.get)
        val labels = InstanceLabels(line._1.labels.map(l => metadata.get.label2Id.getOrElse(l, -1)))
        (labels, instance)
      }
      CrfDataset(labeledInstances, metadata.get)
    }
  }
}