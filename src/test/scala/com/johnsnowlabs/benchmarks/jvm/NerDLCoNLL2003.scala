package com.johnsnowlabs.benchmarks.jvm

import java.io.File
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.benchmarks.spark.NerDLPipeline
import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.ml.tensorflow.{DatasetEncoder, DatasetEncoderParams, TensorflowNer}
import com.johnsnowlabs.nlp.{AnnotatorType, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, TokenizedSentence}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.datasets.CoNLL
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsIndexer}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.tensorflow.{Graph, Session}

import scala.collection.mutable

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
  val encoder = new DatasetEncoder(embeddings.getEmbeddings, settings)

  val graph = new Graph()
  val session = new Session(graph)

  graph.importGraphDef(Files.readAllBytes(Paths.get("char_cnn_blstm_30_25_100_200.pb")))

  val ner = try {
    val model = new TensorflowNer(session, encoder, 9, Verbose.All)
    for (epoch <- 0 until 10) {
      model.train(trainDataset, 0.2f, 0.05f, 9, 0.5f, epoch, epoch + 1)

      System.out.println("\n\nQuality on train data")
      measure(model, trainDataset)

      System.out.println("\n\nQuality on test A data")
      measure(model, testDatasetA)

      System.out.println("\n\nQuality on test B data")
      measure(model, testDatasetB)
    }
    model
  }
  catch {
    case e: Exception =>
      session.close()
      graph.close()
      throw e
  }


  def measure(ner: TensorflowNer, dataset: Array[(TextSentenceLabels, TokenizedSentence)]): Unit = {

    val started = System.nanoTime()

    val tokenized = dataset.map(l => l._2)
    val correctLabels = dataset.map(l => l._1)
    val predictedLabels = ner.predict(tokenized)

    val predictedCorrect = mutable.Map[String, Int]()
    val predicted = mutable.Map[String, Int]()
    val correct = mutable.Map[String, Int]()

    correctLabels.zip(predictedLabels).foreach{
      case (labels, taggedSentence) =>
        labels.labels.zip(taggedSentence).foreach {
          case (label, tag) =>
            correct(label) = correct.getOrElse(label, 0) + 1
            predicted(tag) = predicted.getOrElse(tag, 0) + 1

            if (label == tag)
              predictedCorrect(tag) = predictedCorrect.getOrElse(tag, 0) + 1
        }
    }

    System.out.println(s"time: ${(System.nanoTime() - started)/1e9}")

    val labels = (correct.keys ++ predicted.keys).toSeq.distinct

    val notEmptyLabels = labels.filter(label => label != "O" && label.nonEmpty)

    val totalCorrect = correct.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalPredicted = predicted.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalPredictedCorrect = predictedCorrect.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val (prec, rec, f1) = NerDLPipeline.calcStat(totalCorrect, totalPredicted, totalPredictedCorrect)
    System.out.println(s"Total stat, prec: $prec\t, rec: $rec\t, f1: $f1")

    System.out.println("label\tprec\trec\tf1")

    for (label <- notEmptyLabels) {
      val (prec, rec, f1) = NerDLPipeline.calcStat(
        correct.getOrElse(label, 0),
        predicted.getOrElse(label, 0),
        predictedCorrect.getOrElse(label, 0)
      )

      System.out.println(s"$label\t$prec\t$rec\t$f1")
    }

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