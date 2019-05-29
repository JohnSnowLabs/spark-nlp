package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.TensorResources.extractFloats
import com.johnsnowlabs.nlp.annotators.ner.Verbose

class TensorflowSpell(
  val tensorflow: TensorflowWrapper,
  val verboseLevel: Verbose.Value
  ) extends Logging with Serializable {

  val testInitOp = "test/init"
  val validWords = "valid_words"
  val fileNameTest = "file_name"
  val inMemoryInput = "in-memory-input"
  val batchesKey = "batches"
  val lossKey = "Add:0"
  val dropoutRate = "dropout_rate"

  // these are the inputs to the graph
  val wordIds = "batches:0"
  val contextIds = "batches:1"
  val contextWordIds = "batches:2"

  /* returns the loss associated with the last word, given previous history  */
  def predict(dataset: Array[Array[Int]], cids: Array[Array[Int]], cwids:Array[Array[Int]], configProtoBytes: Option[Array[Byte]] = None) = {

    val tensors = new TensorResources

    val lossWords = tensorflow.getSession(configProtoBytes=configProtoBytes).runner
      .feed(dropoutRate, tensors.createTensor(1.0f))
      .feed(wordIds, tensors.createTensor(dataset.map(_.dropRight(1))))
      .feed(contextIds, tensors.createTensor(cids.map(_.tail)))
      .feed(contextWordIds, tensors.createTensor(cwids.map(_.tail)))
      .fetch(lossKey)
      .fetch(validWords)
      .run()

    tensors.clearTensors()

    val result = extractFloats(lossWords.get(0))
    val width = dataset.head.length
    result.grouped(width - 1).map(_.last)
  }
}
