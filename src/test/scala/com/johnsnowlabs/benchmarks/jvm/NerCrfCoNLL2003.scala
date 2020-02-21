package com.johnsnowlabs.benchmarks.jvm

import com.johnsnowlabs.ml.crf.{CrfParams, LinearChainCrf, LinearChainCrfModel}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.training.CoNLL2003NerReader
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}

import scala.collection.mutable


/*
  Before running:
  1. Download CoNLLL2003 datasets
  2. Set trainFile, testFileA, testFileB to corresponding paths
  3. (Optional) If you wish to use word embeddings then download GLove Word embeddings and unzip it

  Then script could be run
 */
object NerCrfCoNLL2003 extends App {
  val folder = "./"

  val trainFile = ExternalResource(folder + "eng.train", ReadAs.TEXT, Map.empty[String, String])
  val testFileA = ExternalResource(folder + "eng.testa", ReadAs.TEXT, Map.empty[String, String])
  val testFileB = ExternalResource(folder + "eng.testb", ReadAs.TEXT, Map.empty[String, String])

  val embeddingsDims = 100
  val embeddingsFile = folder + s"glove.6B.${embeddingsDims}d.txt"

  val reader = new CoNLL2003NerReader(
    embeddingsFile,
    embeddingsDims,
    normalize=false, // Not normalize might give weight to People names due to upper case, although may overfit
    ReadAs.TEXT,
    Some(ExternalResource("src/test/resources/ner-corpus/dict.txt", ReadAs.TEXT, Map.empty[String, String]))
  )

  def trainModel(er: ExternalResource): LinearChainCrfModel = {
    System.out.println("Dataset Reading")
    val time = System.nanoTime()
    val dataset = reader.readNerDataset(er)

    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")

    System.out.println("Start fitting")

    val params = CrfParams(
      maxEpochs = 10,
      l2 = 1f,
      verbose = Verbose.Epochs,
      randomSeed = Some(0),
      c0 = 2250000
    )
    val crf = new LinearChainCrf(params)
    crf.trainSGD(dataset)
  }

  def testDataset(er: ExternalResource, model: LinearChainCrfModel): Unit = {
    // prec = predicted * correct / predicted
    // rec = predicted * correct / correct
    val started = System.nanoTime()

    val predictedCorrect = mutable.Map[String, Int]()
    val predicted = mutable.    Map[String, Int]()
    val correct = mutable.Map[String, Int]()

    val testDataset = reader.readNerDataset(er, Some(model.metadata))

    for ((labels, sentence) <- testDataset.instances) {

      val predictedLabels = model.predict(sentence)
        .labels
        .map(l => model.metadata.labels(l))
      val correctLabels =
        labels.labels.map{l => if (l >= 0) model.metadata.labels(l) else "unknown"}

      for ((lCorrect, lPredicted) <- correctLabels.zip(predictedLabels)) {
        correct(lCorrect) = correct.getOrElseUpdate(lCorrect, 0) + 1
        predicted(lPredicted) = predicted.getOrElse(lPredicted, 0) + 1

        if (lCorrect == lPredicted)
          predictedCorrect(lPredicted) = predictedCorrect.getOrElseUpdate(lPredicted, 0) + 1
      }
    }

    System.out.println(s"time: ${(System.nanoTime() - started)/1e9}")
    System.out.println("label\tprec\trec\tf1")

    val totalCorrect = correct.filterKeys(label => label != "O").values.sum
    val totalPredicted = predicted.filterKeys(label => label != "O").values.sum
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

  var model = trainModel(trainFile)

  System.out.println("\n\nQuality on train data")
  testDataset(trainFile, model)

  System.out.println("\n\nQuality on test A data")
  testDataset(testFileA, model)

  System.out.println("\n\nQuality on test B data")
  testDataset(testFileB, model)

  for (w <- Array(0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)) {
    System.out.println("w: " + w)
    model = model.shrink(w.toFloat)

    System.out.println("\n\nQuality on train data")
    testDataset(trainFile, model)

    System.out.println("\n\nQuality on test A data")
    testDataset(testFileA, model)

    System.out.println("\n\nQuality on test B data")
    testDataset(testFileB, model)
  }
}
