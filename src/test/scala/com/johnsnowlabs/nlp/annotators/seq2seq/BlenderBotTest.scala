package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.ai.seq2seq.BlenderBot
import org.scalatest.flatspec.AnyFlatSpec

class BlenderBotTest extends AnyFlatSpec {

  //TODO: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/blenderbot#overview
  //Tokenizer Import: https://colab.research.google.com/drive/1Yq9sM8wobn46qfPMwPuqhPUiNfJpfipM?usp=sharing
  //Model Import: https://colab.research.google.com/drive/1nB1lkxOxzujmQ7rCcoy6LvQnqGSm8M9b?usp=sharing
  //Python Inference (Top-k Sampling): https://colab.research.google.com/drive/128fuAOqwmxPehcpIgbkMuDT_imi3bDhc?usp=sharing
  //Python Inference (Beam Search): https://colab.research.google.com/drive/1-wfXev34bPTLG-k8zJfcQ0G2Gd4AVSGd?usp=sharing
  val mainModelPath = "/media/danilo/Data/Danilo/JSL/models/transformers"

  it should "generate BlenderTokenizer using BlenderBot" in {
    val blenderBot = new BlenderBot
    val text = "My friends are cool but they eat too many carbs."
    blenderBot.tag("Hello")
  }

  it should "decode" in {
    val blenderBot = new BlenderBot
    blenderBot.testOutput()
  }

}
