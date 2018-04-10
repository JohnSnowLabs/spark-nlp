package com.johnsnowlabs.benchmarks.jvm

import java.io.File
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.{AnnotatorType, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, TokenizedSentence}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.datasets.CoNLL
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsIndexer}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.nlp.{AnnotatorType, SparkAccessor}
import org.tensorflow.{Graph, Session}

object NerDLCoNLL2003 extends App {
  val spark = SparkAccessor.spark

  val trainFile = ExternalResource("eng.train", ReadAs.LINE_BY_LINE, Map.empty[String, String])
  val testFileA = ExternalResource("eng.testa", ReadAs.LINE_BY_LINE, Map.empty[String, String])
  val testFileB = ExternalResource("eng.testb", ReadAs.LINE_BY_LINE, Map.empty[String, String])

  val wordEmbeddignsFile = "glove.6B.100d.txt"
  val wordEmbeddingsCache = "glove_100_cache.db"
  val wordEmbeddingsDim = 100
  if (!new File(wordEmbeddingsCache).exists())
    WordEmbeddingsIndexer.indexText(wordEmbeddignsFile, wordEmbeddingsCache)

  val embeddings = WordEmbeddings(wordEmbeddingsCache, wordEmbeddingsDim)

  val reader = CoNLL(annotatorType = AnnotatorType.NAMED_ENTITY)
  val trainDataset = toTrain(reader.readDocs(trainFile))
  val testDatasetA = toTrain(reader.readDocs(testFileA))
  val testDatasetB = toTrain(reader.readDocs(testFileB))

  val tags = trainDataset.flatMap(s => s._1.labels).distinct
  val chars = trainDataset.flatMap(s => s._2.tokens.flatMap(t => t.toCharArray)).distinct

  val settings = new DatasetEncoderParams(tags.toList, chars.toList)
  val encoder = new NerDatasetEncoder(embeddings.getEmbeddings, settings)

  val graph = new Graph()
  val session = new Session(graph)

  graph.importGraphDef(Files.readAllBytes(Paths.get("src/main/resources/ner-dl/char_cnn_blstm_10_100_100_25_30.pb")))

  val tf = new TensorflowWrapper(session, graph)

  val ner = try {
    val model = new TensorflowNer(tf, encoder, 9, Verbose.All)
    for (epoch <- 0 until 150) {
      model.train(trainDataset, 0.2f, 0.05f, 9, 0.5f, epoch, epoch + 1)

      System.out.println("\n\nQuality on train data")
      model.measure(trainDataset, (s: String) => System.out.println(s))

      System.out.println("\n\nQuality on test A data")
      model.measure(testDatasetA, (s: String) => System.out.println(s))

      System.out.println("\n\nQuality on test B data")
      model.measure(testDatasetB, (s: String) => System.out.println(s))
    }
    model
  }
  catch {
    case e: Exception =>
      session.close()
      graph.close()
      throw e
  }

  def toTrain(source: Seq[(String, Seq[NerTaggedSentence])]): Array[(TextSentenceLabels, TokenizedSentence)] = {
    source.flatMap{s =>
      s._2.map { sentence =>
        val tokenized = TokenizedSentence(sentence.indexedTaggedWords.map(t => IndexedToken(t.word, t.begin, t.end)))
        val labels = TextSentenceLabels(sentence.tags)

        (labels, tokenized)
      }
    }.toArray
  }
}
