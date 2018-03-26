package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.embeddings.EmbeddingsReadable
import org.apache.spark.ml.util.DefaultParamsReadable

object annotator {

  type Tokenizer = com.johnsnowlabs.nlp.annotators.Tokenizer
  object Tokenizer extends DefaultParamsReadable[Tokenizer]

  type Normalizer = com.johnsnowlabs.nlp.annotators.Normalizer
  object Normalizer extends DefaultParamsReadable[Normalizer]

  type DateMatcher = com.johnsnowlabs.nlp.annotators.DateMatcher
  object DateMatcher extends DefaultParamsReadable[DateMatcher]

  type EntityExtractor = com.johnsnowlabs.nlp.annotators.EntityExtractor
  object EntityExtractor extends DefaultParamsReadable[EntityExtractor]
  type EntityExtractorModel = com.johnsnowlabs.nlp.annotators.EntityExtractorModel
  object EntityExtractorModel extends ParamsAndFeaturesReadable[EntityExtractorModel]

  type RegexMatcher = com.johnsnowlabs.nlp.annotators.RegexMatcher
  object RegexMatcher extends DefaultParamsReadable[RegexMatcher]
  type RegexMatcherModel = com.johnsnowlabs.nlp.annotators.RegexMatcherModel
  object RegexMatcherModel extends ParamsAndFeaturesReadable[RegexMatcherModel]

  type Stemmer = com.johnsnowlabs.nlp.annotators.Stemmer
  object Stemmer extends DefaultParamsReadable[Stemmer]

  type AssertionLogRegApproach = com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegApproach
  object AssertionLogRegApproach extends DefaultParamsReadable[AssertionLogRegApproach]
  type AssertionLogRegModel = com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegModel
  object AssertionLogRegModel extends EmbeddingsReadable[AssertionLogRegModel]

  type NerCrfApproach = com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach
  object NerCrfApproach extends DefaultParamsReadable[NerCrfApproach]
  type NerCrfModel = com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
  object NerCrfModel extends EmbeddingsReadable[NerCrfModel]

  type DependencyParserApproach = com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach
  object DependencyParserApproach extends DefaultParamsReadable[DependencyParserApproach]
  type DependencyParserModel = com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
  object DependencyParserModel extends DefaultParamsReadable[DependencyParserModel]

  type PerceptronApproach = com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach
  object PerceptronApproach extends DefaultParamsReadable[PerceptronApproach]
  type PerceptronModel = com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
  object PerceptronModel extends ParamsAndFeaturesReadable[PerceptronModel]

  type SentenceDetector = com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  object SentenceDetector extends DefaultParamsReadable[SentenceDetector]

  type SentimentDetector = com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector
  object SentimentDetector extends DefaultParamsReadable[SentimentDetector]
  type SentimentDetectorModel = com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel
  object SentimentDetectorModel extends ParamsAndFeaturesReadable[SentimentDetectorModel]

  type ViveknSentimentApproach = com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
  object ViveknSentimentApproach extends DefaultParamsReadable[ViveknSentimentApproach]
  type ViveknSentimentModel = com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel
  object ViveknSentimentModel extends ParamsAndFeaturesReadable[ViveknSentimentModel]

  type NorvigSweetingApproach = com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
  object NorvigSweetingApproach extends DefaultParamsReadable[NorvigSweetingApproach]
  type NorvigSweetingModel = com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
  object NorvigSweetingModel extends ParamsAndFeaturesReadable[NorvigSweetingModel]

}
