package com.johnsnowlabs.downloader.models

import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.downloader.ResourceDownloader
import com.johnsnowlabs.nlp.annotator.{NorvigSweetingModel, ViveknSentimentModel}
import com.johnsnowlabs.nlp.annotators.LemmatizerModel

object CloudPerceptronModel {
  def retrieve = ResourceDownloader.downloadModel(PerceptronModel, "pos_fast", Some("en"))
}

object CloudNerCrfModel {
  def retrieve = ResourceDownloader.downloadModel(NerCrfModel, "ner_fast", Some("en"))
}

object CloudLemmatizer {
  def retrieve = ResourceDownloader.downloadModel(LemmatizerModel, "lemma_fast", Some("en"))
}

object CloudSpellChecker {
  def retrieve = ResourceDownloader.downloadModel(NorvigSweetingModel, "spell_fast", Some("en"))
}

object CloudViveknSentiment {
  def retrieve = ResourceDownloader.downloadModel(ViveknSentimentModel, "vivekn_fast", Some("en"))
}

