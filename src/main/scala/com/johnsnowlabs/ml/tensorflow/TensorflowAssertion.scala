package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.datasets.AssertionAnnotationWithLabel
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import com.johnsnowlabs.ml.tensorflow.TensorResources._

/**
  * Created by jose on 15/03/18.
  */

class TensorflowAssertion (
  val tensorflow: TensorflowWrapper,
  val encoder: AssertionDatasetEncoder,
  val batchSize: Int,
  val verboseLevel: Verbose.Value
) extends Logging {

  private val sentenceLengthsKey = "word_repr/sentence_lengths"
  private val wordEmbeddingsKey = "word_repr/word_embeddings"
  private val labelsKey = "training/labels"
  private val dropoutKey = "training/dropout"
  private val learningRateKey = "training/lr"
  private val trainingKey = "training/Adam"
  private val lossKey = "training/loss"
  private val outputKey = "output_label"

  def predict(dataset: Array[Array[String]], start:Array[Int], end:Array[Int]): Array[String] = {

    val result = ArrayBuffer[String]()

    for ((sents, start, end)
         <- (dataset.grouped(batchSize).toList, start.grouped(batchSize).toList, end.grouped(batchSize).toList).zipped) {

      val batchInput = encoder.encodeInputData(sents, start, end)
      val tensors = new TensorResources()

      val output = tensorflow.session.runner
        .feed(sentenceLengthsKey, tensors.createTensor(batchInput.sentenceLengths))
        .feed(wordEmbeddingsKey, tensors.createTensor(batchInput.wordEmbeddings))
        .fetch(outputKey)
        .run()

      tensors.clearTensors()

      val tagIds = extractInts(output.get(0), batchSize)
      val tags = encoder.decodeOutputData(tagIds)

      result.appendAll(tags)
    }
    result.toArray
  }

  def train(trainDataset: Array[(Array[String], AssertionAnnotationWithLabel)],
            lr: Float,
            batchSize: Int,
            dropout: Float,
            startEpoch: Int,
            endEpoch: Int
           ): Unit = {

    log(s"Training started, trainExamples: ${trainDataset.length} ", Verbose.TrainingStat)

    // Initialize
    if (startEpoch == 0)
      tensorflow.session.runner.addTarget("init").run()

    // initial value for learning rate
    var learningRate = lr

    // Train
    for (epoch <- startEpoch until endEpoch) {

      val epochDataset = Random.shuffle(trainDataset.toList).toArray
      log(s"Epoch: $epoch started, learning rate: $learningRate, dataset size: ${epochDataset.length}", Verbose.Epochs)

      val time = System.nanoTime()
      var batches = 0
      var loss = 0f

      for (batchData <- epochDataset.grouped(batchSize)) {

        val (sentences, annotations) = batchData.unzip
        val labels = annotations.map(r => encoder.encodeOneHot(r.label))

        val start = annotations.map(_.start)
        val end = annotations.map(_.end)
        val batchInput = encoder.encodeInputData(sentences, start, end)

        val tensors = new TensorResources()
        val calculated = tensorflow.session.runner
          .feed(sentenceLengthsKey, tensors.createTensor(batchInput.sentenceLengths))
          .feed(wordEmbeddingsKey, tensors.createTensor(batchInput.wordEmbeddings))
          .feed(labelsKey, tensors.createTensor(labels))

          .feed(dropoutKey, tensors.createTensor(1.0f - dropout))
          .feed(learningRateKey, tensors.createTensor(learningRate))

          .fetch(lossKey)
          .addTarget(trainingKey)
          .run()

        loss += calculated.get(0).floatValue()

        tensors.clearTensors()
        batches += 1
      }

      learningRate = learningRate * 0.95f
      System.out.println(s"Done, ${(System.nanoTime() - time)/1e9} loss: $loss, batches: $batches")
    }
  }
}
