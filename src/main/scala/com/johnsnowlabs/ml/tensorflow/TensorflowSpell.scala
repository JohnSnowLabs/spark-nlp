package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.TensorResources.extractFloats
import com.johnsnowlabs.nlp.annotators.ner.Verbose

class TensorflowSpell(
  val tensorflow: TensorflowWrapper,
  val verboseLevel: Verbose.Value
  ) extends Logging with Serializable {

  val testInitOp = "test/init"
  val validWords = "valid_words"
  val lossKey = "Add:0"
  val dropoutRate = "dropout_rate"

  // these are the inputs to the graph
  val wordIds = "batches:0"
  val contextIds = "batches:1"
  val contextWordIds = "batches:2"

  val testWids = "test_wids"
  val testCids = "test_cids"
  val losses = "test_losses"

  /* returns the loss associated with the last word, given previous history  */
  def predict(dataset: Array[Array[Int]], cids: Array[Array[Int]], cwids:Array[Array[Int]], configProtoBytes: Option[Array[Byte]] = None) = {

    val tensors = new TensorResources

    val lossWords = tensorflow.getSession(configProtoBytes=configProtoBytes).runner
      .feed(dropoutRate, tensors.createTensor(1.0f))
      . feed(wordIds, tensors.createTensor(dataset.map(_.dropRight(1))))
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


  def predict_(dataset: Array[Array[Int]], cids: Array[Array[Int]], cwids: Array[Array[Int]],
               candCids:Array[Int], candWids:Array[Int],
               configProtoBytes: Option[Array[Byte]] = None) = {

    val tensors = new TensorResources
    val paths = (dataset, cids, cwids).zipped.toList

    paths.flatMap { case (pathIds, pathCids, pathWids) =>
      val lossWords = tensorflow.getSession(configProtoBytes = configProtoBytes).runner
        .feed(dropoutRate, tensors.createTensor(1.0f))
        .feed(wordIds, tensors.createTensor(Array(pathIds)))
        .feed(contextIds, tensors.createTensor(Array(pathCids.tail)))
        .feed(contextWordIds, tensors.createTensor(Array(pathWids.tail)))
        .feed(testCids, tensors.createTensor(Array(candCids)))
        .feed(testWids, tensors.createTensor(Array(candWids)))
        .fetch(losses)
        .run()

      tensors.clearTensors()
      val r = extractFloats(lossWords.get(0))
      r
    }
  }
}
