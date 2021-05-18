/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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
import com.johnsnowlabs.nlp.annotators.btm.ReadablePretrainedBigTextMatcher
import com.johnsnowlabs.nlp.annotators.classifier.dl.{ReadClassifierDLTensorflowModel, ReadMultiClassifierDLTensorflowModel, ReadSentimentDLTensorflowModel, ReadablePretrainedClassifierDL, ReadablePretrainedMultiClassifierDL, ReadablePretrainedSentimentDL}
import com.johnsnowlabs.nlp.annotators.{ReadablePretrainedLemmatizer, ReadablePretrainedStopWordsCleanerModel, ReadablePretrainedTextMatcher, ReadablePretrainedTokenizer}
import com.johnsnowlabs.nlp.annotators.ner.crf.ReadablePretrainedNerCrf
import com.johnsnowlabs.nlp.annotators.ner.dl.{ReadablePretrainedNerDL, ReadsNERGraph, WithGraphResolver}
import com.johnsnowlabs.nlp.annotators.parser.dep.ReadablePretrainedDependency
import com.johnsnowlabs.nlp.annotators.parser.typdep.ReadablePretrainedTypedDependency
import com.johnsnowlabs.nlp.annotators.pos.perceptron.ReadablePretrainedPerceptron
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ReadablePretrainedVivekn
import com.johnsnowlabs.nlp.annotators.spell.norvig.ReadablePretrainedNorvig
import com.johnsnowlabs.nlp.annotators.spell.symmetric.ReadablePretrainedSymmetric
import com.johnsnowlabs.nlp.embeddings._
import com.johnsnowlabs.nlp.annotators.ld.dl.{ReadLanguageDetectorDLTensorflowModel, ReadablePretrainedLanguageDetectorDLModel}
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.{ReadablePretrainedSentenceDetectorDL, ReadsSentenceDetectorDLGraph}
import com.johnsnowlabs.nlp.annotators.seq2seq.{ReadMarianMTTensorflowModel, ReadablePretrainedMarianMTModel, ReadT5TransformerTensorflowModel, ReadablePretrainedT5TransformerModel}
import com.johnsnowlabs.nlp.annotators.ws.ReadablePretrainedWordSegmenter

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
  object StopWordsCleaner extends DefaultParamsReadable[StopWordsCleaner] with ReadablePretrainedStopWordsCleanerModel

  type NGramGenerator = com.johnsnowlabs.nlp.annotators.NGramGenerator
  object NGramGenerator extends DefaultParamsReadable[NGramGenerator]

  type NerCrfApproach = com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach
  object NerCrfApproach extends DefaultParamsReadable[NerCrfApproach]
  type NerCrfModel = com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
  object NerCrfModel extends ReadablePretrainedNerCrf

  type PerceptronApproach = com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach
  object PerceptronApproach extends DefaultParamsReadable[PerceptronApproach]
  type PerceptronApproachDistributed = com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproachDistributed
  object PerceptronApproachDistributed extends DefaultParamsReadable[PerceptronApproachDistributed]
  type PerceptronModel = com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
  object PerceptronModel extends ReadablePretrainedPerceptron

  type SentenceDetector = com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  object SentenceDetector extends DefaultParamsReadable[SentenceDetector]

  type SentimentDetector = com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector
  object SentimentDetector extends DefaultParamsReadable[SentimentDetector]
  type SentimentDetectorModel = com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel
  object SentimentDetectorModel extends ParamsAndFeaturesReadable[SentimentDetectorModel]

  type ViveknSentimentApproach = com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
  object ViveknSentimentApproach extends DefaultParamsReadable[ViveknSentimentApproach]
  type ViveknSentimentModel = com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel
  object ViveknSentimentModel extends ReadablePretrainedVivekn

  type NorvigSweetingApproach = com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
  object NorvigSweetingApproach extends DefaultParamsReadable[NorvigSweetingApproach]
  type NorvigSweetingModel = com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
  object NorvigSweetingModel extends ReadablePretrainedNorvig

  type SymmetricDeleteApproach = com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach
  object SymmetricDeleteApproach extends DefaultParamsReadable[SymmetricDeleteApproach]
  type SymmetricDeleteModel = com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel
  object SymmetricDeleteModel extends ReadablePretrainedSymmetric

  type NerDLApproach = com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
  object NerDLApproach extends DefaultParamsReadable[NerDLApproach] with WithGraphResolver
  type NerDLModel = com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
  object NerDLModel extends ReadablePretrainedNerDL with ReadsNERGraph

  type NerConverter = com.johnsnowlabs.nlp.annotators.ner.NerConverter
  object NerConverter extends ParamsAndFeaturesReadable[NerConverter]

  type DependencyParserApproach = com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach
  object DependencyParserApproach extends DefaultParamsReadable[DependencyParserApproach]
  type DependencyParserModel = com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
  object DependencyParserModel extends ReadablePretrainedDependency

  type TypedDependencyParserApproach = com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach
  object TypedDependencyParserApproach extends DefaultParamsReadable[TypedDependencyParserApproach]
  type TypedDependencyParserModel = com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
  object TypedDependencyParserModel extends ReadablePretrainedTypedDependency

  type WordEmbeddings = com.johnsnowlabs.nlp.embeddings.WordEmbeddings
  object WordEmbeddings extends DefaultParamsReadable[WordEmbeddings]
  type WordEmbeddingsModel = com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
  object WordEmbeddingsModel extends ReadablePretrainedWordEmbeddings with EmbeddingsCoverage

  type BertEmbeddings = com.johnsnowlabs.nlp.embeddings.BertEmbeddings
  object BertEmbeddings extends ReadablePretrainedBertModel with ReadBertTensorflowModel

  type SentenceEmbeddings = com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings
  object SentenceEmbeddings extends DefaultParamsReadable[SentenceEmbeddings]

  type ChunkEmbeddings = com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings
  object ChunkEmbeddings extends DefaultParamsReadable[ChunkEmbeddings]

  type NerOverwriter = com.johnsnowlabs.nlp.annotators.ner.NerOverwriter
  object NerOverwriter extends DefaultParamsReadable[NerOverwriter]

  type UniversalSentenceEncoder = com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
  object UniversalSentenceEncoder extends ReadablePretrainedUSEModel with ReadUSETensorflowModel

  type ElmoEmbeddings = com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings
  object ElmoEmbeddings extends ReadablePretrainedElmoModel with ReadElmoTensorflowModel

  type ClassifierDLApproach = com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
  object ClassifierDLApproach extends DefaultParamsReadable[ClassifierDLApproach]
  type ClassifierDLModel = com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel
  object ClassifierDLModel extends ReadablePretrainedClassifierDL with ReadClassifierDLTensorflowModel

  type AlbertEmbeddings = com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings
  object AlbertEmbeddings extends ReadablePretrainedAlbertModel with ReadAlbertTensorflowModel with ReadSentencePieceModel

  type XlnetEmbeddings = com.johnsnowlabs.nlp.embeddings.XlnetEmbeddings
  object XlnetEmbeddings extends ReadablePretrainedXlnetModel with ReadXlnetTensorflowModel with ReadSentencePieceModel

  type SentimentDLApproach = com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLApproach
  object SentimentDLApproach extends DefaultParamsReadable[SentimentDLApproach]
  type SentimentDLModel = com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel
  object SentimentDLModel extends ReadablePretrainedSentimentDL with ReadSentimentDLTensorflowModel

  type LanguageDetectorDL = com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL
  object LanguageDetectorDL extends ReadablePretrainedLanguageDetectorDLModel with ReadLanguageDetectorDLTensorflowModel

  type BertSentenceEmbeddings = com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings
  object BertSentenceEmbeddings extends ReadablePretrainedBertSentenceModel with ReadBertSentenceTensorflowModel

  type MultiClassifierDLApproach = com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach
  object MultiClassifierDLApproach extends DefaultParamsReadable[MultiClassifierDLApproach]

  type MultiClassifierDLModel = com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLModel
  object MultiClassifierDLModel extends ReadablePretrainedMultiClassifierDL with ReadMultiClassifierDLTensorflowModel

  type SentenceDetectorDLApproach  = com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach
  object SentenceDetectorDLApproach extends DefaultParamsReadable[SentenceDetectorDLApproach]

  type SentenceDetectorDLModel  = com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
  object SentenceDetectorDLModel extends ReadsSentenceDetectorDLGraph with ReadablePretrainedSentenceDetectorDL

  type WordSegmenterApproach = com.johnsnowlabs.nlp.annotators.ws.WordSegmenterApproach
  object WordSegmenterApproach extends DefaultParamsReadable[WordSegmenterApproach]
  type WordSegmenterModel = com.johnsnowlabs.nlp.annotators.ws.WordSegmenterModel
  object WordSegmenterModel extends ReadablePretrainedWordSegmenter

  type DocumentNormalizer = com.johnsnowlabs.nlp.annotators.DocumentNormalizer
  object DocumentNormalizer extends DefaultParamsReadable[DocumentNormalizer]

  type MarianTransformer = com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer
  object MarianTransformer extends ReadablePretrainedMarianMTModel with ReadMarianMTTensorflowModel with ReadSentencePieceModel
  
  type T5Transformer = com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
  object T5Transformer extends ReadablePretrainedT5TransformerModel with ReadT5TransformerTensorflowModel with ReadSentencePieceModel

  type DistilBertEmbeddings = com.johnsnowlabs.nlp.embeddings.DistilBertEmbeddings
  object DistilBertEmbeddings extends ReadablePretrainedDistilBertModel with ReadDistilBertTensorflowModel

  type RoBertaEmbeddings = com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings
  object RoBertaEmbeddings extends ReadablePretrainedRobertaModel with ReadRobertaTensorflowModel

}
