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

  val tensors = new TensorResources()

  /* returns the loss associated with the last word, given previous history  */
  def predict(dataset: Array[Array[Int]], cids: Array[Array[Int]], cwids:Array[Array[Int]]) = this.synchronized {

    val packed = dataset.zip(cids).zip(cwids).map {
      case ((_ids, _cids), _cwids) => Array(_ids, _cids, _cwids)
    }

    val inputTensor = tensors.createTensor(packed)

    tensorflow.session.runner
      .feed(inMemoryInput, inputTensor)
      .addTarget(testInitOp)
      .run()

    val lossWords = tensorflow.session.runner
      .feed(dropoutRate, tensors.createTensor(1.0f))
      .fetch(lossKey)
      .fetch(validWords)
      .run()

    val result = extractFloats(lossWords.get(0))
    val width = inputTensor.shape()(2)
    result.grouped(width.toInt - 1).map(_.last)

  }
}
