package com.johnsnowlabs.ml.pytorch

import ai.djl.Model
import ai.djl.pytorch.engine.PtModel
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import org.scalatest.flatspec.AnyFlatSpec

import java.io.ByteArrayInputStream

class PytorchBertTest extends AnyFlatSpec with SparkSessionTest {

  behavior of "PytorchBertTest"

  it should "infer" in {
    val torchModelPath = "tmp_bert_base_cased_pt/bert_pytorch"
    val tokens = Array(101, 2627, 1108, 3104, 1124, 15703, 136, 102, 103, 1124, 15703, 1108, 170, 16797, 8284, 102)
    val segments = Array(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)
    val input = Array(tokens, segments)

    val pytorchWrapper = PytorchWrapper(torchModelPath)
    val pytorchBert = new PytorchBert(pytorchWrapper, 1, 1)
    val output = pytorchBert.tag(input)

    println(output.head.head.mkString(" "))
    assert(output.length == 2)
    assert(output.head.length == 16)
    assert(output.head.head.length == 768)
  }

  it should "read" in {
    val torchModelPath = "pytorch_models/bert_pytorch"
    val pytorchWrapper = PytorchWrapper(torchModelPath)
    val modelInputStream = new ByteArrayInputStream(pytorchWrapper.modelBytes)

    val pyTorchModel: PtModel = Model.newInstance("bert_pytorch").asInstanceOf[PtModel]
    pyTorchModel.load(modelInputStream)
  }

}
