package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.TensorResources.extractFloats
import com.johnsnowlabs.nlp.annotators.ner.Verbose

import scala.collection.mutable.ArrayBuffer

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

  /* TODO: hard-coded stuff for test */
  val sentMatrix = tensors.createTensor(Array(
    Array(1, 8008, 3358, 4902, 5324, 3008, 845, 2),
    Array(1, 8008, 9663, 4902, 5324, 3008, 845, 2)))

  val test = tensorflow.session.runner
    .feed(inMemoryInput, sentMatrix)
    .addTarget(testInitOp)
    .run()


  def predict(dataset: Array[Array[Int]]): Array[Float] = {
    val inputTensor = tensors.createTensor(dataset)

    val loss = tensorflow.session.runner
      .feed(inMemoryInput, inputTensor)
      .feed(dropoutRate, tensors.createTensor(1.0f))
      .addTarget(testInitOp)
      .fetch(lossKey)
      .run()

    val tagIds = extractFloats(loss.get(0),loss.get(0).numElements)
    tagIds
  }

}
