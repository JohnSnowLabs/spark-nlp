/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp

import com.johnsnowlabs.ml.tensorflow.sentencepiece.ReadSentencePieceModel
import com.johnsnowlabs.nlp.annotators.audio._
import com.johnsnowlabs.nlp.annotators.btm.ReadablePretrainedBigTextMatcher
import com.johnsnowlabs.nlp.annotators.classifier.dl._
import com.johnsnowlabs.nlp.annotators.coref.{
  ReadSpanBertCorefTensorflowModel,
  ReadablePretrainedSpanBertCorefModel
}
import com.johnsnowlabs.nlp.annotators.cv._
import com.johnsnowlabs.nlp.annotators.er.ReadablePretrainedEntityRuler
import com.johnsnowlabs.nlp.annotators.ld.dl.{
  ReadLanguageDetectorDLTensorflowModel,
  ReadablePretrainedLanguageDetectorDLModel
}
import com.johnsnowlabs.nlp.annotators.ner.crf.ReadablePretrainedNerCrf
import com.johnsnowlabs.nlp.annotators.ner.dl._
import com.johnsnowlabs.nlp.annotators.parser.dep.ReadablePretrainedDependency
import com.johnsnowlabs.nlp.annotators.parser.typdep.ReadablePretrainedTypedDependency
import com.johnsnowlabs.nlp.annotators.pos.perceptron.ReadablePretrainedPerceptron
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ReadablePretrainedVivekn
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.{
  ReadablePretrainedSentenceDetectorDL,
  ReadsSentenceDetectorDLGraph
}
import com.johnsnowlabs.nlp.annotators.seq2seq._
import com.johnsnowlabs.nlp.annotators.spell.norvig.ReadablePretrainedNorvig
import com.johnsnowlabs.nlp.annotators.spell.symmetric.ReadablePretrainedSymmetric
import com.johnsnowlabs.nlp.annotators.ws.ReadablePretrainedWordSegmenter
import com.johnsnowlabs.nlp.annotators.{
  ReadablePretrainedLemmatizer,
  ReadablePretrainedStopWordsCleanerModel,
  ReadablePretrainedTextMatcher,
  ReadablePretrainedTokenizer
}
import com.johnsnowlabs.nlp.embeddings._
import org.apache.spark.ml.util.DefaultParamsReadable

package object annotator {

  type Tokenizer = com.johnsnowlabs.nlp.annotators.Tokenizer

  object Tokenizer extends DefaultParamsReadable[Tokenizer]

  type TokenizerModel = com.johnsnowlabs.nlp.annotators.TokenizerModel

  object TokenizerModel extends ReadablePretrainedTokenizer

  type RegexTokenizer = com.johnsnowlabs.nlp.annotators.RegexTokenizer

  object RegexTokenizer extends DefaultParamsReadable[RegexTokenizer]

  type RecursiveTokenizer = com.johnsnowlabs.nlp.annotators.RecursiveTokenizer

  object RecursiveTokenizer extends DefaultParamsReadable[RecursiveTokenizer]

  type RecursiveTokenizerModel = com.johnsnowlabs.nlp.annotators.RecursiveTokenizerModel

  object RecursiveTokenizerModel extends ReadablePretrainedTokenizer

  type ChunkTokenizer = com.johnsnowlabs.nlp.annotators.ChunkTokenizer

  object ChunkTokenizer extends DefaultParamsReadable[ChunkTokenizer]

  type Token2Chunk = com.johnsnowlabs.nlp.annotators.Token2Chunk

  object Token2Chunk extends DefaultParamsReadable[Token2Chunk]

  type Normalizer = com.johnsnowlabs.nlp.annotators.Normalizer

  object Normalizer extends DefaultParamsReadable[Normalizer]

  type NormalizerModel = com.johnsnowlabs.nlp.annotators.NormalizerModel

  object NormalizerModel extends ParamsAndFeaturesReadable[NormalizerModel]

  type DateMatcher = com.johnsnowlabs.nlp.annotators.DateMatcher

  object DateMatcher extends DefaultParamsReadable[DateMatcher]

  type MultiDateMatcher = com.johnsnowlabs.nlp.annotators.MultiDateMatcher

  object MultiDateMatcher extends DefaultParamsReadable[MultiDateMatcher]

  type TextMatcher = com.johnsnowlabs.nlp.annotators.TextMatcher

  object TextMatcher extends DefaultParamsReadable[TextMatcher]

  type TextMatcherModel = com.johnsnowlabs.nlp.annotators.TextMatcherModel

  object TextMatcherModel extends ReadablePretrainedTextMatcher

  type BigTextMatcher = com.johnsnowlabs.nlp.annotators.btm.BigTextMatcher

  object BigTextMatcher extends DefaultParamsReadable[BigTextMatcher]

  type BigTextMatcherModel = com.johnsnowlabs.nlp.annotators.btm.BigTextMatcherModel

  object BigTextMatcherModel extends ReadablePretrainedBigTextMatcher

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

  object LemmatizerModel extends ReadablePretrainedLemmatizer

  type StopWordsCleaner = com.johnsnowlabs.nlp.annotators.StopWordsCleaner

  object StopWordsCleaner
      extends DefaultParamsReadable[StopWordsCleaner]
      with ReadablePretrainedStopWordsCleanerModel

  type NGramGenerator = com.johnsnowlabs.nlp.annotators.NGramGenerator

  object NGramGenerator extends DefaultParamsReadable[NGramGenerator]

  type NerCrfApproach = com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach

  object NerCrfApproach extends DefaultParamsReadable[NerCrfApproach]

  type NerCrfModel = com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel

  object NerCrfModel extends ReadablePretrainedNerCrf

  type PerceptronApproach = com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach

  object PerceptronApproach extends DefaultParamsReadable[PerceptronApproach]

  type PerceptronApproachDistributed =
    com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproachDistributed

  object PerceptronApproachDistributed
      extends DefaultParamsReadable[PerceptronApproachDistributed]

  type PerceptronModel = com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel

  object PerceptronModel extends ReadablePretrainedPerceptron

  type SentenceDetector = com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector

  object SentenceDetector extends DefaultParamsReadable[SentenceDetector]

  type SentimentDetector = com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector

  object SentimentDetector extends DefaultParamsReadable[SentimentDetector]

  type SentimentDetectorModel =
    com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel

  object SentimentDetectorModel extends ParamsAndFeaturesReadable[SentimentDetectorModel]

  type ViveknSentimentApproach =
    com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach

  object ViveknSentimentApproach extends DefaultParamsReadable[ViveknSentimentApproach]

  type ViveknSentimentModel = com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel

  object ViveknSentimentModel extends ReadablePretrainedVivekn

  type NorvigSweetingApproach =
    com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach

  object NorvigSweetingApproach extends DefaultParamsReadable[NorvigSweetingApproach]

  type NorvigSweetingModel = com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel

  object NorvigSweetingModel extends ReadablePretrainedNorvig

  type SymmetricDeleteApproach =
    com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach

  object SymmetricDeleteApproach extends DefaultParamsReadable[SymmetricDeleteApproach]

  type SymmetricDeleteModel = com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel

  object SymmetricDeleteModel extends ReadablePretrainedSymmetric

  type NerDLApproach = com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach

  object NerDLApproach extends DefaultParamsReadable[NerDLApproach] with WithGraphResolver

  type NerDLModel = com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel

  object NerDLModel extends ReadablePretrainedNerDL with ReadsNERGraph

  type NerConverter = com.johnsnowlabs.nlp.annotators.ner.NerConverter

  object NerConverter extends ParamsAndFeaturesReadable[NerConverter]

  type DependencyParserApproach =
    com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach

  object DependencyParserApproach extends DefaultParamsReadable[DependencyParserApproach]

  type DependencyParserModel = com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel

  object DependencyParserModel extends ReadablePretrainedDependency

  type TypedDependencyParserApproach =
    com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach

  object TypedDependencyParserApproach
      extends DefaultParamsReadable[TypedDependencyParserApproach]

  type TypedDependencyParserModel =
    com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel

  object TypedDependencyParserModel extends ReadablePretrainedTypedDependency

  type WordEmbeddings = com.johnsnowlabs.nlp.embeddings.WordEmbeddings

  object WordEmbeddings extends DefaultParamsReadable[WordEmbeddings]

  type WordEmbeddingsModel = com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel

  object WordEmbeddingsModel extends ReadablePretrainedWordEmbeddings with EmbeddingsCoverage

  type BertEmbeddings = com.johnsnowlabs.nlp.embeddings.BertEmbeddings

  object BertEmbeddings extends ReadablePretrainedBertModel with ReadBertDLModel

  type SentenceEmbeddings = com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings

  object SentenceEmbeddings extends DefaultParamsReadable[SentenceEmbeddings]

  type ChunkEmbeddings = com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings

  object ChunkEmbeddings extends DefaultParamsReadable[ChunkEmbeddings]

  type NerOverwriter = com.johnsnowlabs.nlp.annotators.ner.NerOverwriter

  object NerOverwriter extends DefaultParamsReadable[NerOverwriter]

  type UniversalSentenceEncoder = com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder

  object UniversalSentenceEncoder extends ReadablePretrainedUSEModel with ReadUSEDLModel

  type ElmoEmbeddings = com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings

  object ElmoEmbeddings extends ReadablePretrainedElmoModel with ReadElmoDLModel

  type ClassifierDLApproach = com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach

  object ClassifierDLApproach extends DefaultParamsReadable[ClassifierDLApproach]

  type ClassifierDLModel = com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel

  object ClassifierDLModel
      extends ReadablePretrainedClassifierDL
      with ReadClassifierDLTensorflowModel

  type AlbertEmbeddings = com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings

  object AlbertEmbeddings extends ReadablePretrainedAlbertModel with ReadAlbertDLModel

  type XlnetEmbeddings = com.johnsnowlabs.nlp.embeddings.XlnetEmbeddings

  object XlnetEmbeddings extends ReadablePretrainedXlnetModel with ReadXlnetDLModel

  type SentimentDLApproach = com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLApproach

  object SentimentDLApproach extends DefaultParamsReadable[SentimentDLApproach]

  type SentimentDLModel = com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel

  object SentimentDLModel
      extends ReadablePretrainedSentimentDL
      with ReadSentimentDLTensorflowModel

  type YakeKeywordExtraction = com.johnsnowlabs.nlp.annotators.keyword.yake.YakeKeywordExtraction

  object YakeKeywordExtraction extends ParamsAndFeaturesReadable[YakeKeywordExtraction]

  type LanguageDetectorDL = com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL

  object LanguageDetectorDL
      extends ReadablePretrainedLanguageDetectorDLModel
      with ReadLanguageDetectorDLTensorflowModel

  type BertSentenceEmbeddings = com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings

  object BertSentenceEmbeddings
      extends ReadablePretrainedBertSentenceModel
      with ReadBertSentenceDLModel

  type MultiClassifierDLApproach =
    com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach

  object MultiClassifierDLApproach extends DefaultParamsReadable[MultiClassifierDLApproach]

  type MultiClassifierDLModel =
    com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLModel

  object MultiClassifierDLModel
      extends ReadablePretrainedMultiClassifierDL
      with ReadMultiClassifierDLTensorflowModel

  type SentenceDetectorDLApproach =
    com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach

  object SentenceDetectorDLApproach extends DefaultParamsReadable[SentenceDetectorDLApproach]

  type SentenceDetectorDLModel =
    com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel

  object SentenceDetectorDLModel
      extends ReadsSentenceDetectorDLGraph
      with ReadablePretrainedSentenceDetectorDL

  type WordSegmenterApproach = com.johnsnowlabs.nlp.annotators.ws.WordSegmenterApproach

  object WordSegmenterApproach extends DefaultParamsReadable[WordSegmenterApproach]

  type WordSegmenterModel = com.johnsnowlabs.nlp.annotators.ws.WordSegmenterModel

  object WordSegmenterModel extends ReadablePretrainedWordSegmenter

  type DocumentNormalizer = com.johnsnowlabs.nlp.annotators.DocumentNormalizer

  object DocumentNormalizer extends DefaultParamsReadable[DocumentNormalizer]

  type MarianTransformer = com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer

  object MarianTransformer
      extends ReadablePretrainedMarianMTModel
      with ReadMarianMTDLModel
      with ReadSentencePieceModel

  type T5Transformer = com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer

  object T5Transformer
      extends ReadablePretrainedT5TransformerModel
      with ReadT5TransformerDLModel
      with ReadSentencePieceModel

  type DistilBertEmbeddings = com.johnsnowlabs.nlp.embeddings.DistilBertEmbeddings

  object DistilBertEmbeddings extends ReadablePretrainedDistilBertModel with ReadDistilBertDLModel

  type RoBertaEmbeddings = com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings

  object RoBertaEmbeddings extends ReadablePretrainedRobertaModel with ReadRobertaDLModel

  type XlmRoBertaEmbeddings = com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings

  object XlmRoBertaEmbeddings extends ReadablePretrainedXlmRobertaModel with ReadXlmRobertaDLModel

  type BertForTokenClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.BertForTokenClassification

  object BertForTokenClassification
      extends ReadablePretrainedBertForTokenModel
      with ReadBertForTokenDLModel

  type DistilBertForTokenClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForTokenClassification

  object DistilBertForTokenClassification
      extends ReadablePretrainedDistilBertForTokenModel
      with ReadDistilBertForTokenDLModel

  type LongformerEmbeddings = com.johnsnowlabs.nlp.embeddings.LongformerEmbeddings

  object LongformerEmbeddings extends ReadablePretrainedLongformerModel with ReadLongformerDLModel

  type RoBertaSentenceEmbeddings = com.johnsnowlabs.nlp.embeddings.RoBertaSentenceEmbeddings

  object RoBertaSentenceEmbeddings
      extends ReadablePretrainedRobertaSentenceModel
      with ReadRobertaSentenceDLModel

  type XlmRoBertaSentenceEmbeddings = com.johnsnowlabs.nlp.embeddings.XlmRoBertaSentenceEmbeddings

  object XlmRoBertaSentenceEmbeddings
      extends ReadablePretrainedXlmRobertaSentenceModel
      with ReadXlmRobertaSentenceDLModel

  type RoBertaForTokenClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForTokenClassification

  object RoBertaForTokenClassification
      extends ReadablePretrainedRoBertaForTokenModel
      with ReadRoBertaForTokenDLModel

  type XlmRoBertaForTokenClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForTokenClassification

  object XlmRoBertaForTokenClassification
      extends ReadablePretrainedXlmRoBertaForTokenModel
      with ReadXlmRoBertaForTokenDLModel

  type AlbertForTokenClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.AlbertForTokenClassification

  object AlbertForTokenClassification
      extends ReadablePretrainedAlbertForTokenModel
      with ReadAlbertForTokenDLModel

  type XlnetForTokenClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.XlnetForTokenClassification

  object XlnetForTokenClassification
      extends ReadablePretrainedXlnetForTokenModel
      with ReadXlnetForTokenDLModel

  type LongformerForTokenClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.LongformerForTokenClassification

  object LongformerForTokenClassification
      extends ReadablePretrainedLongformerForTokenModel
      with ReadLongformerForTokenDLModel

  type EntityRulerApproach = com.johnsnowlabs.nlp.annotators.er.EntityRulerApproach

  type EntityRulerModel = com.johnsnowlabs.nlp.annotators.er.EntityRulerModel

  object EntityRulerModel extends ReadablePretrainedEntityRuler

  type BertForSequenceClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.BertForSequenceClassification

  object BertForSequenceClassification
      extends ReadablePretrainedBertForSequenceModel
      with ReadBertForSequenceDLModel

  type Doc2VecApproach = com.johnsnowlabs.nlp.embeddings.Doc2VecApproach

  object Doc2VecApproach extends DefaultParamsReadable[Doc2VecApproach]

  type Doc2VecModel = com.johnsnowlabs.nlp.embeddings.Doc2VecModel

  object Doc2VecModel extends ReadablePretrainedDoc2Vec

  type DistilBertForSequenceClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForSequenceClassification

  object DistilBertForSequenceClassification
      extends ReadablePretrainedDistilBertForSequenceModel
      with ReadDistilBertForSequenceDLModel

  type RoBertaForSequenceClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForSequenceClassification

  object RoBertaForSequenceClassification
      extends ReadablePretrainedRoBertaForSequenceModel
      with ReadRoBertaForSequenceDLModel

  type XlmRoBertaForSequenceClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForSequenceClassification

  object XlmRoBertaForSequenceClassification
      extends ReadablePretrainedXlmRoBertaForSequenceModel
      with ReadXlmRoBertaForSequenceDLModel

  type LongformerForSequenceClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.LongformerForSequenceClassification

  object LongformerForSequenceClassification
      extends ReadablePretrainedLongformerForSequenceModel
      with ReadLongformerForSequenceDLModel

  type AlbertForSequenceClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.AlbertForSequenceClassification

  object AlbertForSequenceClassification
      extends ReadablePretrainedAlbertForSequenceModel
      with ReadAlbertForSequenceDLModel

  type XlnetForSequenceClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.XlnetForSequenceClassification

  object XlnetForSequenceClassification
      extends ReadablePretrainedXlnetForSequenceModel
      with ReadXlnetForSequenceDLModel

  type GPT2Transformer = com.johnsnowlabs.nlp.annotators.seq2seq.GPT2Transformer

  object GPT2Transformer
      extends ReadablePretrainedGPT2TransformerModel
      with ReadGPT2TransformerDLModel

  type Word2VecApproach = com.johnsnowlabs.nlp.embeddings.Word2VecApproach

  object Word2VecApproach extends DefaultParamsReadable[Word2VecApproach]

  type Word2VecModel = com.johnsnowlabs.nlp.embeddings.Word2VecModel

  object Word2VecModel extends ReadablePretrainedWord2Vec

  type DeBertaEmbeddings = com.johnsnowlabs.nlp.embeddings.DeBertaEmbeddings

  object DeBertaEmbeddings extends ReadablePretrainedDeBertaModel with ReadDeBertaDLModel

  type DeBertaForSequenceClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.DeBertaForSequenceClassification

  object DeBertaForSequenceClassification
      extends ReadablePretrainedDeBertaForSequenceModel
      with ReadDeBertaForSequenceDLModel

  type DeBertaForTokenClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.DeBertaForTokenClassification

  object DeBertaForTokenClassification
      extends ReadablePretrainedDeBertaForTokenModel
      with ReadDeBertaForTokenDLModel

  type CamemBertEmbeddings = com.johnsnowlabs.nlp.embeddings.CamemBertEmbeddings

  object CamemBertEmbeddings extends ReadablePretrainedCamemBertModel with ReadCamemBertDLModel

  type SpanBertCorefModel = com.johnsnowlabs.nlp.annotators.coref.SpanBertCorefModel

  object SpanBertCorefModel
      extends ReadablePretrainedSpanBertCorefModel
      with ReadSpanBertCorefTensorflowModel

  type BertForQuestionAnswering =
    com.johnsnowlabs.nlp.annotators.classifier.dl.BertForQuestionAnswering

  object BertForQuestionAnswering
      extends ReadablePretrainedBertForQAModel
      with ReadBertForQuestionAnsweringDLModel

  type DistilBertForQuestionAnswering =
    com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForQuestionAnswering

  object DistilBertForQuestionAnswering
      extends ReadablePretrainedDistilBertForQAModel
      with ReadDistilBertForQuestionAnsweringDLModel

  type RoBertaForQuestionAnswering =
    com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForQuestionAnswering

  object RoBertaForQuestionAnswering
      extends ReadablePretrainedRoBertaForQAModel
      with ReadRoBertaForQuestionAnsweringDLModel

  type XlmRoBertaForQuestionAnswering =
    com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForQuestionAnswering

  object XlmRoBertaForQuestionAnswering
      extends ReadablePretrainedXlmRoBertaForQAModel
      with ReadXlmRoBertaForQuestionAnsweringDLModel

  type DeBertaForQuestionAnswering =
    com.johnsnowlabs.nlp.annotators.classifier.dl.DeBertaForQuestionAnswering

  object DeBertaForQuestionAnswering
      extends ReadablePretrainedDeBertaForQAModel
      with ReadDeBertaForQuestionAnsweringDLModel

  type AlbertForQuestionAnswering =
    com.johnsnowlabs.nlp.annotators.classifier.dl.AlbertForQuestionAnswering

  object AlbertForQuestionAnswering
      extends ReadablePretrainedAlbertForQAModel
      with ReadAlbertForQuestionAnsweringDLModel

  type LongformerForQuestionAnswering =
    com.johnsnowlabs.nlp.annotators.classifier.dl.LongformerForQuestionAnswering

  object LongformerForQuestionAnswering
      extends ReadablePretrainedLongformerForQAModel
      with ReadLongformerForQuestionAnsweringDLModel

  type ViTForImageClassification =
    com.johnsnowlabs.nlp.annotators.cv.ViTForImageClassification

  object ViTForImageClassification
      extends ReadablePretrainedViTForImageModel
      with ReadViTForImageDLModel

  type VisionEncoderDecoderForImageCaptioning =
    com.johnsnowlabs.nlp.annotators.cv.VisionEncoderDecoderForImageCaptioning

  object VisionEncoderDecoderForImageCaptioning
      extends ReadablePretrainedVisionEncoderDecoderModel
      with ReadVisionEncoderDecoderDLModel

  type CamemBertForTokenClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.CamemBertForTokenClassification

  object CamemBertForTokenClassification
      extends ReadablePretrainedCamemBertForTokenModel
      with ReadCamemBertForTokenDLModel

  type TapasForQuestionAnswering =
    com.johnsnowlabs.nlp.annotators.classifier.dl.TapasForQuestionAnswering

  object TapasForQuestionAnswering
      extends ReadablePretrainedTapasForQAModel
      with ReadTapasForQuestionAnsweringDLModel

  type CamemBertForSequenceClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.CamemBertForSequenceClassification

  object CamemBertForSequenceClassification
      extends ReadablePretrainedCamemBertForSequenceModel
      with ReadCamemBertForSequenceDLModel

  type SwinForImageClassification =
    com.johnsnowlabs.nlp.annotators.cv.SwinForImageClassification

  object SwinForImageClassification
      extends ReadablePretrainedSwinForImageModel
      with ReadSwinForImageDLModel

  type ConvNextForImageClassification =
    com.johnsnowlabs.nlp.annotators.cv.ConvNextForImageClassification

  object ConvNextForImageClassification
      extends ReadablePretrainedConvNextForImageModel
      with ReadConvNextForImageDLModel

  type CamemBertForQuestionAnswering =
    com.johnsnowlabs.nlp.annotators.classifier.dl.CamemBertForQuestionAnswering

  object CamemBertForQuestionAnswering
      extends ReadablePretrainedCamemBertForQAModel
      with ReadCamemBertForQADLModel

  type Wav2Vec2ForCTC =
    com.johnsnowlabs.nlp.annotators.audio.Wav2Vec2ForCTC

  object Wav2Vec2ForCTC
      extends ReadablePretrainedWav2Vec2ForAudioModel
      with ReadWav2Vec2ForAudioDLModel

  type HubertForCTC =
    com.johnsnowlabs.nlp.annotators.audio.HubertForCTC

  object HubertForCTC extends ReadablePretrainedHubertForAudioModel with ReadHubertForAudioDLModel

  type WhisperForCTC =
    com.johnsnowlabs.nlp.annotators.audio.WhisperForCTC

  object WhisperForCTC extends ReadablePretrainedWhisperForCTCModel with ReadWhisperForCTCDLModel

  type ZeroShotNerModel =
    com.johnsnowlabs.nlp.annotators.ner.dl.ZeroShotNerModel

  object ZeroShotNerModel extends ReadablePretrainedZeroShotNer with ReadZeroShotNerDLModel

  type Date2Chunk = com.johnsnowlabs.nlp.annotators.Date2Chunk

  object Date2Chunk extends DefaultParamsReadable[Date2Chunk]

  type Chunk2Doc = com.johnsnowlabs.nlp.annotators.Chunk2Doc

  object Chunk2Doc extends DefaultParamsReadable[Chunk2Doc]

  type BartTransformer = com.johnsnowlabs.nlp.annotators.seq2seq.BartTransformer

  object BartTransformer
      extends ReadablePretrainedBartTransformerModel
      with ReadBartTransformerDLModel

  type BertForZeroShotClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.BertForZeroShotClassification

  object BertForZeroShotClassification
      extends ReadablePretrainedBertForZeroShotModel
      with ReadBertForZeroShotDLModel

  type DistilBertForZeroShotClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForZeroShotClassification

  object DistilBertForZeroShotClassification
      extends ReadablePretrainedDistilBertForZeroShotModel
      with ReadDistilBertForZeroShotDLModel

  type RobertaBertForZeroShotClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForZeroShotClassification

  object RoBertaForZeroShotClassification
      extends ReadablePretrainedRoBertaForZeroShotModel
      with ReadRoBertaForZeroShotDLModel

  type XlmRobertaBertForZeroShotClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForZeroShotClassification

  object XlmRoBertaForZeroShotClassification
      extends ReadablePretrainedXlmRoBertaForZeroShotModel
      with ReadXlmRoBertaForZeroShotDLModel

  type E5Embeddings = com.johnsnowlabs.nlp.embeddings.E5Embeddings

  object E5Embeddings extends ReadablePretrainedE5Model with ReadE5DLModel

  type InstructorEmbeddings = com.johnsnowlabs.nlp.embeddings.InstructorEmbeddings

  object InstructorEmbeddings extends ReadablePretrainedInstructorModel with ReadInstructorDLModel

  type BartForZeroShotClassification =
    com.johnsnowlabs.nlp.annotators.classifier.dl.BartForZeroShotClassification

  object BartForZeroShotClassification
      extends ReadablePretrainedBartForZeroShotModel
      with ReadBartForZeroShotDLModel

  type DocumentCharacterTextSplitter =
    com.johnsnowlabs.nlp.annotators.DocumentCharacterTextSplitter
  object DocumentCharacterTextSplitter
      extends ParamsAndFeaturesReadable[DocumentCharacterTextSplitter]

  type DocumentTokenSplitter =
    com.johnsnowlabs.nlp.annotators.DocumentTokenSplitter

  object DocumentTokenSplitter extends ParamsAndFeaturesReadable[DocumentTokenSplitter]
}
