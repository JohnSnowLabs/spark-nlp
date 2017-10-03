package com.jsl.ml.crf

/*
  Before running:
  1. Download CoNLLL2003 datasets
  2. Set trainFile, testFileA, testFileB to corresponding paths

  Then script could be run
 */
object CoNLL2003Test extends App {
  val folder = "/home/aleksei/work/nlp/libs/crfsuite_exp/conll2003/"

  val trainFile = folder + "eng.train.crfsuite"
  val testFileA = folder + "eng.testa.crfsuite"
  val testFileB = folder + "eng.testb.crfsuite"


  def trainModel(file: String, linesToSkip: Int): LinearChainCrfModel = {
    System.out.println("Dataset Reading")
    val time = System.nanoTime()
    val dataset = DatasetReader.readAndEncode(trainFile, linesToSkip)
    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")

    System.out.println("Start fitting")

    val params = TrainParams(
      minEpochs = 100,
      l2 = 1f,
      verbose = Verbose.Epochs,
      randomSeed = Some(0),
      c0 = 2250000
    )

    val crf = new LinearChainCrf(params)
    crf.trainSGD(dataset)
  }

  def testDataset(file: String, linesToSkip: Int, model: LinearChainCrfModel, metadata: DatasetMetadata): Unit = {
    // prec = predicted * correct / predicted
    // rec = predicted * correct / correct
    val started = System.nanoTime()

    val labels = metadata.label2Id.size
    val predictedCorrect = Array.fill(labels)(0)
    val predicted = Array.fill(labels)(0)
    val correct = Array.fill(labels)(0)

    val testInstances = DatasetReader.readAndEncode(file, linesToSkip, metadata)
    for ((labels, instance) <- testInstances) {
      val predictedLabels = model.predict(instance)
      for ((lCorrect, lPredicted) <- labels.labels.zip(predictedLabels.labels)
           if lCorrect >= 0) {

        correct(lCorrect) += 1
        predicted(lPredicted) += 1

        if (lCorrect == lPredicted)
          predictedCorrect(lPredicted) += 1
      }
    }

    System.out.println(s"time: ${(System.nanoTime() - started)/1e9}")
    System.out.println("label\tprec\trec\tf1")

    for (i <- 1 until labels) {
      val label = metadata.label2Id.filter(p => p._2 == i).keys.head
      val rec = predictedCorrect(i).toFloat / correct(i)
      val prec = predictedCorrect(i).toFloat / predicted(i)
      val f1 = 2 * prec * rec / (prec + rec)

      System.out.println(s"$label\t$prec\t$rec\t$f1")
    }
  }

  val model = trainModel(trainFile, 2)

  System.out.println("\n\nQuality on train data")
  testDataset(trainFile, 2, model, model.metadata)

  System.out.println("\n\nQuality on test A data")
  testDataset(testFileA, 2, model, model.metadata)

  System.out.println("\n\nQuality on test B data")
  testDataset(testFileB, 2, model, model.metadata)

}

