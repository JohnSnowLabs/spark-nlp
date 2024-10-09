package com.johnsnowlabs.debug

import org.scalatest.flatspec.AnyFlatSpec

class GenericTransformerTest extends AnyFlatSpec {

  "Some Transformer" should "work" in {
    val question = "Water boils at what temperature?"
    val result = GenericTransformer.tokenizedWithSentence(question)
    println("")
  }


  it should "load ONNX model" in {
    val question = "Water boils at what temperature?" //i.e. context, prompt
    val text = "100°C"
    GenericTransformer.predictWithContext(text, question)
  }

  it should "work for multiple choice" in {
    val prompt = "The Great Wall of China was built to protect against invasions from which group?"
    val choices = Array("The Greeks", "The Romans", "The Mongols", "The Persians")

    GenericTransformer.predictWithContextAndMultipleChoice(prompt, choices)
  }

}
