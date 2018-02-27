package com.johnsnowlabs.pretrained.en.models

import com.johnsnowlabs.nlp
import com.johnsnowlabs.nlp.annotators
import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.pretrained.ResourceDownloader

object S3DocumentAssembler {
  def retrieveStandard = ResourceDownloader.downloadModel(nlp.DocumentAssembler, "document_std", Some("en"))
}

object S3SentenceDetector {
  def retrieveStandard = ResourceDownloader.downloadModel(annotators.sbd.pragmatic.SentenceDetector, "sentence_std", Some("en"))
}

object S3Tokenizer {
  def retrieveStandard = ResourceDownloader.downloadModel(annotators.Tokenizer, "tokenizer_std", Some("en"))
}

object S3PerceptronModel {
  def retrieveSmall = ResourceDownloader.downloadModel(PerceptronModel, "pos_fast", Some("en"))
}

object S3NerCrfModel {
  def retrieveSmall = ResourceDownloader.downloadModel(NerCrfModel, "ner_fast", Some("en"))
}

