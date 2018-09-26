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
  val batchesKey = "batches"
  val lossKey = "loss/loss"
  val dropoutRate = "dropout_rate"

  val tensors = new TensorResources()

  tensorflow.session.runner
    .feed(fileNameTest, tensors.createTensor("../auxdata/data/gap_filling_exercise.ids"))
    .addTarget(testInitOp)
    .run()


  def predict(dataset: Array[Array[String]], start:Array[Int], end:Array[Int]): Array[Float] = {

    val result = ArrayBuffer[String]()

    val loss = tensorflow.session.runner
        .feed(dropoutRate, tensors.createTensor(1.0f))
        .fetch(lossKey)
        .run()

    val tagIds = extractFloats(loss.get(0),loss.get(0).numElements)
    tagIds
  }

}
