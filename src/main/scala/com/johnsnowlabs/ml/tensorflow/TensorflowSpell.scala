package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.TensorResources.extractFloats
import com.johnsnowlabs.nlp.annotators.ner.Verbose

class TensorflowSpell(
  val tensorflow: TensorflowWrapper,
  val verboseLevel: Verbose.Value
  ) extends Logging {

  val testInitOp = "test/init"
  val validWords = "valid_words"
  val fileNameTest = "file_name"
  val inMemoryInput = "in-memory-input"
  val batchesKey = "batches"
  val lossKey = "loss/loss"
  val dropoutRate = "dropout_rate"

  val tensors = new TensorResources()

  /* returns the loss associated with the last word */
  def predict(dataset: Array[Array[Int]]) = {
    val inputTensor = tensors.createTensor(dataset)

    tensorflow.session.runner
      .feed(inMemoryInput, inputTensor)
      .addTarget(testInitOp)
      .run()

    val lossWords = tensorflow.session.runner
      .feed(dropoutRate, tensors.createTensor(1.0f))
      .fetch(lossKey)
      .fetch(validWords)
      .run()

    val result = extractFloats(lossWords.get(0),lossWords.get(0).numElements)
    val width = inputTensor.shape()(1)
    result.grouped(width.toInt - 1).map(_.last)

  }
}
