package com.johnsnowlabs.ml.crf

import java.io.File

import com.johnsnowlabs.nlp.annotators.common.{TaggedSentence, WordEmbeddings, WordEmbeddingsIndexer}
import com.johnsnowlabs.nlp.annotators.ner.crf.{DictionaryFeatures, FeatureGenerator}
import com.johnsnowlabs.nlp.AnnotatorType
import com.johnsnowlabs.nlp.datasets.CoNLL

import scala.collection.mutable


/*
  Before running:
  1. Download CoNLLL2003 datasets
  2. Set trainFile, testFileA, testFileB to corresponding paths
  3. (Optional) If you wish to use word embeddings then download GLove Word embeddings and unzip it

  Then script could be run
 */
object CoNLL2003CrfTest extends App {
  val folder = "./"

  val trainFile = folder + "eng.train"
  val testFileA = folder + "eng.testa"
  val testFileB = folder + "eng.testb"

  val embeddingsDims = 100
  val embeddingsFile = folder + s"glove.6B.${embeddingsDims}d.txt"
  val wordEmbeddingsDb = folder + s"embeddings.${embeddingsDims}d.db"

  var wordEmbeddings: Option[WordEmbeddings] = None

  val time = System.nanoTime()
  if (new File(embeddingsFile).exists() && !new File(wordEmbeddingsDb).exists()) {
    WordEmbeddingsIndexer.indexGloveToLevelDb(embeddingsFile, wordEmbeddingsDb)
  }

  if (new File(wordEmbeddingsDb).exists()) {
    wordEmbeddings = Some(WordEmbeddings(wordEmbeddingsDb, embeddingsDims))
  }

  val nerReader = CoNLL(3, AnnotatorType.NAMED_ENTITY)
  val posReader = CoNLL(1, AnnotatorType.POS)
  val fg = FeatureGenerator(
    DictionaryFeatures.read(Seq("src/main/resources/ner-corpus/dict.txt")),
    wordEmbeddings
  )

  def readDataset(file: String): Seq[(TextSentenceLabels, TaggedSentence)] = {
    val labels = nerReader.readDocs(file).flatMap(_._2)
      .map(sentence => TextSentenceLabels(sentence.tags))

    val posTaggedSentences = posReader.readDocs(file).flatMap(_._2)
    labels.zip(posTaggedSentences)
  }

  def trainModel(file: String): LinearChainCrfModel = {
    System.out.println("Dataset Reading")
    val time = System.nanoTime()
    val lines = readDataset(file)
    val dataset = fg.generateDataset(lines)
    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")

    System.out.println("Start fitting")

    val params = CrfParams(
      maxEpochs = 10,
      l2 = 1f,
      verbose = Verbose.Epochs,
      randomSeed = Some(0),
      c0 = 1250000
    )
    val crf = new LinearChainCrf(params)
    crf.trainSGD(dataset)
  }

  def testDataset(file: String, model: LinearChainCrfModel): Unit = {
    // prec = predicted * correct / predicted
    // rec = predicted * correct / correct
    val started = System.nanoTime()

    val predictedCorrect = mutable.Map[String, Int]()
    val predicted = mutable.Map[String, Int]()
    val correct = mutable.Map[String, Int]()

    val testInstances = readDataset(file)

    for ((labels, sentence) <- testInstances) {
      val instance = fg.generate(sentence, model.metadata)

      val predictedLabels = model.predict(instance)
        .labels
        .map(l => model.metadata.labels(l))

      for ((lCorrect, lPredicted) <- labels.labels.zip(predictedLabels)) {
        correct(lCorrect) = correct.getOrElseUpdate(lCorrect, 0) + 1
        predicted(lPredicted) = predicted.getOrElse(lPredicted, 0) + 1

        if (lCorrect == lPredicted)
          predictedCorrect(lPredicted) = predictedCorrect.getOrElseUpdate(lPredicted, 0) + 1
      }
    }

    System.out.println(s"time: ${(System.nanoTime() - started)/1e9}")
    System.out.println("label\tprec\trec\tf1")

    val totalCorrect = correct.filterKeys(label => label != "O").values.sum
    val totalPredicted = correct.filterKeys(label => label != "O").values.sum
    val totalPredictedCorrect = predictedCorrect.filterKeys(label => label != "O").values.sum

    val rec = totalPredictedCorrect.toFloat / totalCorrect
    val prec = totalPredictedCorrect.toFloat / totalPredicted
    val f1 = 2 * prec * rec / (prec + rec)

    System.out.println(s"Total\t$prec\t$rec\t$f1")

    val labels = (predicted.keys ++ correct.keys).toList.distinct

    for (label <- labels) {
      val rec = predictedCorrect.getOrElse(label, 0).toFloat / correct.getOrElse(label, 0)
      val prec = predictedCorrect.getOrElse(label, 0).toFloat / predicted.getOrElse(label, 0)
      val f1 = 2 * prec * rec / (prec + rec)

      System.out.println(s"$label\t$prec\t$rec\t$f1")
    }
  }

  val model = trainModel(trainFile)

  System.out.println("\n\nQuality on train data")
  testDataset(trainFile, model)

  System.out.println("\n\nQuality on test A data")
  testDataset(testFileA, model)

  System.out.println("\n\nQuality on test B data")
  testDataset(testFileB, model)
}

