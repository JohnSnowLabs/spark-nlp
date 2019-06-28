package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.{PretrainedLemmatizer, PretrainedTokenizer}
import com.johnsnowlabs.nlp.annotators.ner.crf.PretrainedNerCrf
import com.johnsnowlabs.nlp.annotators.ner.dl.{PretrainedNerDL, ReadsNERGraph, WithGraphResolver}
import com.johnsnowlabs.nlp.annotators.parser.dep.PretrainedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.PretrainedTypedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PretrainedPerceptronModel
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknPretrainedModel
import com.johnsnowlabs.nlp.annotators.spell.context.{PretrainedSpellModel, ReadsLanguageModelGraph}
import com.johnsnowlabs.nlp.annotators.spell.norvig.PretrainedNorvigSweeting
import com.johnsnowlabs.nlp.annotators.spell.symmetric.PretrainedSymmetricDelete
import com.johnsnowlabs.nlp.embeddings.{EmbeddingsReadable, PretrainedBertModel, PretrainedWordEmbeddings, ReadBertTensorflowModel}
import org.apache.spark.ml.util.DefaultParamsReadable

package object annotator {

  type Tokenizer = com.johnsnowlabs.nlp.annotators.Tokenizer
  object Tokenizer extends DefaultParamsReadable[Tokenizer] with PretrainedTokenizer

  type ChunkTokenizer = com.johnsnowlabs.nlp.annotators.ChunkTokenizer
  object ChunkTokenizer extends DefaultParamsReadable[ChunkTokenizer]

  type Normalizer = com.johnsnowlabs.nlp.annotators.Normalizer
  object Normalizer extends DefaultParamsReadable[Normalizer]
  type NormalizerModel = com.johnsnowlabs.nlp.annotators.NormalizerModel
  object NormalizerModel extends ParamsAndFeaturesReadable[NormalizerModel]

  type DateMatcher = com.johnsnowlabs.nlp.annotators.DateMatcher
  object DateMatcher extends DefaultParamsReadable[DateMatcher]

  type TextMatcher = com.johnsnowlabs.nlp.annotators.TextMatcher
  object TextMatcher extends DefaultParamsReadable[TextMatcher]
  type TextMatcherModel = com.johnsnowlabs.nlp.annotators.TextMatcherModel
  object TextMatcherModel extends ParamsAndFeaturesReadable[TextMatcherModel]

  type RegexMatcher = com.johnsnowlabs.nlp.annotators.RegexMatcher
  object RegexMatcher extends DefaultParamsReadable[RegexMatcher]
  type RegexMatcherModel = com.johnsnowlabs.nlp.annotators.RegexMatcherModel
  object RegexMatcherModel extends ParamsAndFeaturesReadable[RegexMatcherModel]

  type Chunker = com.johnsnowlabs.nlp.annotators.Chunker
  object Chunker extends DefaultParamsReadable[Chunker]

  type Stemmer = com.johnsnowlabs.nlp.annotators.Stemmer
  object Stemmer extends DefaultParamsReadable[Stemmer]

  type Lemmatizer = com.johnsnowlabs.nlp.annotators.Lemmatizer
  object Lemmatizer extends DefaultParamsReadable[Lemmatizer]
  type LemmatizerModel = com.johnsnowlabs.nlp.annotators.LemmatizerModel
  object LemmatizerModel extends ParamsAndFeaturesReadable[LemmatizerModel] with PretrainedLemmatizer

  type NerCrfApproach = com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach
  object NerCrfApproach extends DefaultParamsReadable[NerCrfApproach]
  type NerCrfModel = com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
  object NerCrfModel extends ParamsAndFeaturesReadable[NerCrfModel] with PretrainedNerCrf

  type PerceptronApproach = com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach
  object PerceptronApproach extends DefaultParamsReadable[PerceptronApproach]
  type PerceptronApproachDistributed = com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproachDistributed
  object PerceptronApproachDistributed extends DefaultParamsReadable[PerceptronApproachDistributed]
  type PerceptronModel = com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
  object PerceptronModel extends ParamsAndFeaturesReadable[PerceptronModel] with PretrainedPerceptronModel

  type SentenceDetector = com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  object SentenceDetector extends DefaultParamsReadable[SentenceDetector]

  type DeepSentenceDetector = com.johnsnowlabs.nlp.annotators.sbd.deep.DeepSentenceDetector
  object DeepSentenceDetector extends DefaultParamsReadable[DeepSentenceDetector]

  type SentimentDetector = com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector
  object SentimentDetector extends DefaultParamsReadable[SentimentDetector]
  type SentimentDetectorModel = com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel
  object SentimentDetectorModel extends ParamsAndFeaturesReadable[SentimentDetectorModel]

  type ViveknSentimentApproach = com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
  object ViveknSentimentApproach extends DefaultParamsReadable[ViveknSentimentApproach]
  type ViveknSentimentModel = com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel
  object ViveknSentimentModel extends ParamsAndFeaturesReadable[ViveknSentimentModel] with ViveknPretrainedModel

  type NorvigSweetingApproach = com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
  object NorvigSweetingApproach extends DefaultParamsReadable[NorvigSweetingApproach]
  type NorvigSweetingModel = com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
  object NorvigSweetingModel extends ParamsAndFeaturesReadable[NorvigSweetingModel] with PretrainedNorvigSweeting

  type SymmetricDeleteApproach = com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach
  object SymmetricDeleteApproach extends DefaultParamsReadable[SymmetricDeleteApproach]
  type SymmetricDeleteModel = com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel
  object SymmetricDeleteModel extends ParamsAndFeaturesReadable[SymmetricDeleteModel] with PretrainedSymmetricDelete

  type ContextSpellCheckerApproach = com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerApproach
  object ContextSpellCheckerApproach extends DefaultParamsReadable[ContextSpellCheckerApproach]
  type ContextSpellCheckerModel = com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel
  object ContextSpellCheckerModel extends ReadsLanguageModelGraph with PretrainedSpellModel

  type NerDLApproach = com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
  object NerDLApproach extends DefaultParamsReadable[NerDLApproach] with WithGraphResolver
  type NerDLModel = com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
  object NerDLModel extends ParamsAndFeaturesReadable[NerDLModel] with ReadsNERGraph with PretrainedNerDL

  type NerConverter = com.johnsnowlabs.nlp.annotators.ner.NerConverter
  object NerConverter extends ParamsAndFeaturesReadable[NerConverter]

  type DependencyParserApproach = com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach
  object DependencyParserApproach extends DefaultParamsReadable[DependencyParserApproach]
  type DependencyParserModel = com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
  object DependencyParserModel extends ParamsAndFeaturesReadable[DependencyParserModel] with PretrainedDependencyParserModel

  type TypedDependencyParserApproach = com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach
  object TypedDependencyParserApproach extends DefaultParamsReadable[TypedDependencyParserApproach]
  type TypedDependencyParserModel = com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
  object TypedDependencyParserModel extends ParamsAndFeaturesReadable[TypedDependencyParserModel] with PretrainedTypedDependencyParserModel

  type WordEmbeddings = com.johnsnowlabs.nlp.embeddings.WordEmbeddings
  object WordEmbeddings extends DefaultParamsReadable[WordEmbeddings]
  type WordEmbeddingsModel = com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
  object WordEmbeddingsModel extends EmbeddingsReadable[WordEmbeddingsModel] with PretrainedWordEmbeddings

  type BertEmbeddings = com.johnsnowlabs.nlp.embeddings.BertEmbeddings
  object BertEmbeddings extends ParamsAndFeaturesReadable[BertEmbeddings] with PretrainedBertModel with ReadBertTensorflowModel

}
