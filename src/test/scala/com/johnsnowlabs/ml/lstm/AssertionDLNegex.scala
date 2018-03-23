package com.johnsnowlabs.ml.lstm

import java.io.File
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.ml.common.EvaluationMetrics
import com.johnsnowlabs.ml.logreg.{Datapoint, NegexDatasetReader}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.annotators.datasets.AssertionAnnotationWithLabel
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsIndexer}
import org.tensorflow.{Graph, Session}

import scala.util.Random

/**
  * Train an Assertion Status Model with Deep Learning and the toy negex dataset
  */
object AssertionDLNegex extends App with EvaluationMetrics {


  def randomSplit(dataset:Seq[Datapoint], fraction: Float) = {
    val shuffled = Random.shuffle(dataset)
    val trainSize = (fraction * shuffled.length).toInt
    val testSize = (shuffled.length - trainSize).toInt
    Array(shuffled.take(trainSize), shuffled.takeRight(testSize))
  }

  val datasetPath = "rsAnnotations-1-120-random.txt"

  val mappings = Map("Affirmed" -> 0, "Negated" -> 1)
  val reader = new NegexDatasetReader()

  val ds = reader.readDataset(datasetPath)
  val Array(trainingData, testingData) = randomSplit(ds, 0.7f).map(createDatapoints)

  // word embeddings
  val wordEmbeddingsFile = s"PubMed-shuffle-win-2.bin"
  val wordEmbeddingsCache = s"PubMed-shuffle-win-2.bin.db"
  val wordEmbeddingsDim = 200
  if (!new File(wordEmbeddingsCache).exists())
    WordEmbeddingsIndexer.indexBinary(wordEmbeddingsFile, wordEmbeddingsCache)

  val embeddings = WordEmbeddings(wordEmbeddingsCache, wordEmbeddingsDim)
  val labels = List("Affirmed", "Negated")
  val params = new DatasetEncoderParams(labels, List.empty)

  val encoder = new AssertionDatasetEncoder(embeddings.getEmbeddings, params)
  val graph = new Graph()
  val session = new Session(graph)

  graph.importGraphDef(Files.readAllBytes(Paths.get("./src/test/resources/assertion.lstm/blstm_34_32_30_200.pb")))

  val tf = new TensorflowWrapper(session, graph)
  val batchSize = 16

  val assertion = try {
    val model = new TensorflowAssertion(tf, encoder, batchSize, Verbose.All)
    for (epoch <- 0 until 6) {
      model.train(trainingData, 0.0015f, 16, 0.1f, epoch, epoch + 1)
      System.out.println("Quality on train data")
      measure(model, trainingData)
      System.out.println("Quality on test data")
      measure(model, testingData)
    }
    model
  }
  catch {
    case e: Exception =>
      session.close()
      graph.close()
      throw e
  }

  def createDatapoints(dataset: Seq[Datapoint]):Array[(Array[String], AssertionAnnotationWithLabel)] = dataset.map{ point =>
      (point.sentence.split(" "), AssertionAnnotationWithLabel(point.label, point.start, point.end))
  }.toArray

  /* computes accuracy and prints it on stdout */
  def measure(model: TensorflowAssertion, dataset: Array[(Array[String], AssertionAnnotationWithLabel)]): Unit = {
    val starts = dataset.map(_._2.start)
    val ends = dataset.map(_._2.end)
    val sentences = dataset.map(_._1)
    val labels = dataset.map(_._2.label)
    val predicted = model.predict(sentences, starts, ends)
    val correct = labels.zip(predicted).filter{case (x, y) => x.equals(y)}.size.toDouble
    println(correct / predicted.length)
  }
}
