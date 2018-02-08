package com.johnsnowlabs.pretrained.en.models

import com.johnsnowlabs.nlp.annotators
import com.johnsnowlabs.nlp
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.pretrained.ResourceDownloader


object DocumentAssembler {
  def std = ResourceDownloader.downloadModel(nlp.DocumentAssembler, "document_std", Some("en"))
}

object SentenceDetector {
  def std = ResourceDownloader.downloadModel(annotators.sbd.pragmatic.SentenceDetector, "sentence_std", Some("en"))
}

object Tokenizer {
  def std = ResourceDownloader.downloadModel(annotators.Tokenizer, "tokenizer_std", Some("en"))
}

object Pos {
  def fast = ResourceDownloader.downloadModel(PerceptronModel, "pos_fast", Some("en"))
}

object Ner {
  lazy val fast = ResourceDownloader.downloadModel(PerceptronModel, "ner_fast", Some("en"))
}
