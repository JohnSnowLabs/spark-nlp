package com.johnsnowlabs.benchmarks.jvm

import java.io.File

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.SparkAccessor
import com.johnsnowlabs.nlp.annotators.common.{TokenPieceEmbeddings, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.annotators.ner.dl.{LoadsContrib, NerDLApproach}
import com.johnsnowlabs.nlp.training.{CoNLL, CoNLLDocument}
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddingsReader, WordEmbeddingsTextIndexer, WordEmbeddingsWriter}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.storage.RocksDBConnection
import org.tensorflow.Session


object NerDLCoNLL2003 extends App {

  val spark = SparkAccessor.spark

  val trainFile = ExternalResource("eng.train", ReadAs.TEXT, Map.empty[String, String])
  val testFileA = ExternalResource("eng.testa", ReadAs.TEXT, Map.empty[String, String])
  val testFileB = ExternalResource("eng.testb", ReadAs.TEXT, Map.empty[String, String])

  val wordEmbeddignsFile = "glove.6B.100d.txt"
  val wordEmbeddingsCache = "glove_100_cache.db"
  val wordEmbeddingsDim = 100

  lazy val connection = RocksDBConnection.getOrCreate(wordEmbeddingsCache)

  if (!new File(wordEmbeddingsCache).exists()) {
    WordEmbeddingsTextIndexer.index(wordEmbeddignsFile, new WordEmbeddingsWriter(connection, false, wordEmbeddingsDim, 5000, 5000))
  }

  val embeddings = new WordEmbeddingsReader(connection, false, wordEmbeddingsDim, 1000)

  val reader = CoNLL()
  val trainDataset = toTrain(reader.readDocs(trainFile), embeddings)
  val testDatasetA = toTrain(reader.readDocs(testFileA), embeddings)
  val testDatasetB = toTrain(reader.readDocs(testFileB), embeddings)

  val tags = trainDataset.flatMap(s => s._1.labels).distinct
  val chars = trainDataset.flatMap(s => s._2.tokens.flatMap(t => t.wordpiece.toCharArray)).distinct

  val settings = DatasetEncoderParams(tags.toList, chars.toList,
    Array.fill[Float](wordEmbeddingsDim)(0f).toList, wordEmbeddingsDim)
  val encoder = new NerDatasetEncoder(settings)

  //Use CPU
  //val config = Array[Byte](10, 7, 10, 3, 67, 80, 85, 16, 0)
  //Use GPU
  //val config = Array[Byte](56, 1)
  //val config = Array[Byte](50, 2, 32, 1, 56, 1, 64, 1)
  val config = Array[Byte](50, 2, 32, 1, 56, 1)
  LoadsContrib.loadContribToTensorflow()
  val graph = TensorflowWrapper.readGraph("src/main/resources/ner-dl/blstm_10_100_128_100.pb")
  val session = new Session(graph, config)


  val tf = new TensorflowWrapper(Variables(Array.empty[Byte], Array.empty[Byte]), graph.toGraphDef)


  val ner = try {
    val model = new TensorflowNer(tf, encoder, 32, Verbose.All)
    for (epoch <- 0 until 150) {
      model.train(trainDataset.grouped(32), trainDataset.size, Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]().grouped(32),
        trainDataset.size, 1e-3f, 0.005f, 0.5f, epoch, epoch + 1, outputLogsPath = "")

      System.out.println("\n\nQuality on train data")
      model.measure(trainDataset.grouped(32), extended = true, outputLogsPath = "")

      System.out.println("\n\nQuality on test A data")
      model.measure(testDatasetA.grouped(32), extended = true, outputLogsPath = "")

      System.out.println("\n\nQuality on test B data")
      model.measure(testDatasetB.grouped(32), extended = true, outputLogsPath = "")
    }
    model
  }
  catch {
    case e: Exception =>
      session.close()
      graph.close()
      throw e
  }

  def toTrain(source: Seq[CoNLLDocument], embeddings: WordEmbeddingsReader):
      Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)] = {

    source.flatMap{s =>
      s.nerTagged.zipWithIndex.map { case (sentence, idx) =>
        val tokens = sentence.indexedTaggedWords.map {t =>
          val vectorOption = embeddings.lookup(t.word)
          TokenPieceEmbeddings(t.word, t.word, -1, true, vectorOption, Array.fill[Float](wordEmbeddingsDim)(0f), t.begin, t.end)
        }
        val tokenized = WordpieceEmbeddingsSentence(tokens, idx)
        val labels = TextSentenceLabels(sentence.tags)

        (labels, tokenized)
      }
    }.toArray
  }
}
