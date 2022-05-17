#  Copyright 2017-2022 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from test.annotator import *
from test.annotator.basic_annotator_test_spec import BasicAnnotatorsTestSpec
from test.annotator.chunk_doc_serializing_test_spec import ChunkDocSerializingTestSpec
from test.annotator.chunker_test_spec import ChunkerTestSpec
from test.annotator.classifier_dl.albert_for_token_classification_test_spec import \
    AlbertForTokenClassificationTestSpec
from test.annotator.classifier_dl.bert_for_token_classification_test_spec import \
    BertForTokenClassificationTestSpec
from test.annotator.classifier_dl.classifier_dl_test_spec import ClassifierDLTestSpec
from test.annotator.classifier_dl.deberta_for_sequence_classification_test_spec import \
    DeBertaForSequenceClassificationTestSpec
from test.annotator.classifier_dl.distil_bert_for_sequence_classification_test_spec import \
    DistilBertForSequenceClassificationTestSpec
from test.annotator.classifier_dl.longformer_for_token_classification_test_spec import \
    LongformerForTokenClassificationTestSpec
from test.annotator.classifier_dl.multi_classifier_dl_test_spec import \
    MultiClassifierDLTestSpec
from test.annotator.classifier_dl.roberta_for_sequence_classification_test_spec import \
    RoBertaForSequenceClassificationTestSpec
from test.annotator.classifier_dl.roberta_for_token_classification_test_spec import \
    RoBertaForTokenClassificationTestSpec
from test.annotator.classifier_dl.xlm_roberta_for_sequence_classification_test_spec import \
    XlmRoBertaForSequenceClassificationTestSpec
from test.annotator.classifier_dl.xlm_roberta_for_token_classification_test_spec import \
    XlmRoBertaForTokenClassificationTestSpec
from test.annotator.classifier_dl.xlnet_for_token_classification_test_spec import \
    XlnetForTokenClassificationTestSpec
from test.annotator.dependency.dependency_parser_test_spec import \
    DependencyParserConllUTestSpec, DependencyParserTreeBankTestSpec
from test.annotator.dependency.typed_dependency_parser_test_spec import \
    TypedDependencyParserConllUTestSpec, TypedDependencyParserConll2009TestSpec
from test.annotator.document_normalizer_spec import DocumentNormalizerSpec
from test.annotator.embeddings.albert_embeddings_test_spec import \
    AlbertEmbeddingsTestSpec
from test.annotator.embeddings.camembert_embeddings_test_spec import \
    CamemBertEmbeddingsTestSpec
from test.annotator.embeddings.chunk_embeddings_test_spec import ChunkEmbeddingsTestSpec
from test.annotator.embeddings.deberta_for_token_classification_test_spec import \
    DeBertaForTokenClassificationTestSpec
from test.annotator.embeddings.doc2_vec_test_spec import Doc2VecTestSpec
from test.annotator.embeddings.elmo_embeddings_test_spec import ElmoEmbeddingsTestSpec
from test.annotator.embeddings.embeddings_finisher_test_spec import \
    EmbeddingsFinisherTestSpec
from test.annotator.embeddings.roberta_sentence_embeddings_test_spec import \
    RoBertaSentenceEmbeddingsTestSpec
from test.annotator.embeddings.sentence_embeddings_test_spec import \
    SentenceEmbeddingsTestSpec
from test.annotator.embeddings.universal_sentence_encoder_test_spec import \
    UniversalSentenceEncoderTestSpec
from test.annotator.embeddings.word2_vec_test_spec import Word2VecTestSpec
from test.annotator.embeddings.xlnet_embeddings_test_spec import XlnetEmbeddingsTestSpec
from test.annotator.er.entity_ruler_test_spec import EntityRulerTestSpec
from test.annotator.get_classes_test_spec import GetClassesTestSpec
from test.annotator.keyword_extraction.yake_keyword_extraction_test_spec import \
    YakeKeywordExtractionTestSpec
from test.annotator.ld_dl.language_detector_dl_test_spec import \
    LanguageDetectorDLTestSpec
from test.annotator.lemmatizer_test_spec import LemmatizerWithTrainingDataSetTestSpec, \
    LemmatizerTestSpec
from test.annotator.matcher.date_matcher_test_spec import DateMatcherTestSpec
from test.annotator.matcher.regex_matcher_test_spec import RegexMatcherTestSpec
from test.annotator.matcher.text_matcher_test_spec import TextMatcherTestSpec
from test.annotator.n_gram_generator_test_spec import NGramGeneratorTestSpec
from test.annotator.ner.ner_dl_model_test_spec import NerDLModelTestSpec
from test.annotator.normalizer_test_spec import NormalizerTestSpec
from test.annotator.pos.perceptron_test_spec import PerceptronApproachTestSpec
from test.annotator.sentence.pragmatic_test_spec import PragmaticScorerTestSpec, \
    PragmaticSBDTestSpec
from test.annotator.sentence.sentence_detector_dl_test_spec import \
    SentenceDetectorDLExtraParamsTestSpec, SentenceDetectorDLTestSpec
from test.annotator.sentiment.sentiment_dl_test_spec import SentimentDLTestSpec
from test.annotator.seq2seq.gpt2_transformer_text_generation_test_spec import \
    GPT2TransformerTextGenerationTestSpec
from test.annotator.seq2seq.t5_transformer_test_spec import \
    T5TransformerSummaryWithRepetitionPenaltyTestSpec, \
    T5TransformerSummaryWithSamplingAndTopPTestSpec, \
    T5TransformerSummaryWithSamplingAndTemperatureTestSpec, \
    T5TransformerSummaryWithSamplingAndDeactivatedTopKTestSpec, \
    T5TransformerSummaryWithSamplingTestSpec, T5TransformerSummaryTestSpec, \
    T5TransformerQATestSpec
from test.annotator.spell_check.context_spell_checker_model_test_spec import \
    ContextSpellCheckerModelTestSpec
from test.annotator.spell_check.norvig_sweeting_model_test_spec import \
    NorvigSweetingModelTestSpec
from test.annotator.spell_check.spell_checker_test_spec import SpellCheckerTestSpec
from test.annotator.spell_check.symmetric_delete_test_spec import \
    SymmetricDeleteTestSpec
from test.annotator.stop_words_cleaner_test_spec import StopWordsCleanerModelTestSpec, \
    StopWordsCleanerTestSpec
from test.annotator.token.chunk_tokenizer_test_spec import ChunkTokenizerTestSpec
from test.annotator.token.regex_tokenizer_test_spec import RegexTokenizerTestSpec
from test.annotator.token.tokenizer_test_spec import TokenizerWithExceptionsTestSpec, \
    TokenizerTestSpec
from test.annotator.ws.word_segmenter_test_spec import WordSegmenterTestSpec
from test.functions import FunctionMapColumnsTestSpec, FunctionMapColumnTestSpec
from test.misc import *
from test.base import *
from test.pretrained import *

suite = unittest.TestSuite()

# Annotator tests
suite.addTest(BasicAnnotatorsTestSpec())
suite.addTest(RegexMatcherTestSpec())
suite.addTest(TokenizerTestSpec())
suite.addTest(TokenizerWithExceptionsTestSpec())
suite.addTest(NormalizerTestSpec())
suite.addTest(ChunkTokenizerTestSpec())
suite.addTest(LemmatizerTestSpec())
suite.addTest(LemmatizerWithTrainingDataSetTestSpec())
suite.addTest(DateMatcherTestSpec())
suite.addTest(TextMatcherTestSpec())
suite.addTest(DocumentNormalizerSpec())
suite.addTest(RegexTokenizerTestSpec())

suite.addTest(PerceptronApproachTestSpec())
suite.addTest(ChunkerTestSpec())
suite.addTest(ChunkDocSerializingTestSpec())

suite.addTest(PragmaticSBDTestSpec())
suite.addTest(PragmaticScorerTestSpec())
# suite.addTest(PipelineTestSpec())  # TODO
suite.addTest(SpellCheckerTestSpec())
suite.addTest(NorvigSweetingModelTestSpec())
suite.addTest(SymmetricDeleteTestSpec())
suite.addTest(ContextSpellCheckerModelTestSpec())
# suite.addTest(ParamsGettersTestSpec())  # TODO
suite.addTest(DependencyParserTreeBankTestSpec())
suite.addTest(DependencyParserConllUTestSpec())
suite.addTest(TypedDependencyParserConll2009TestSpec())
suite.addTest(TypedDependencyParserConllUTestSpec())
suite.addTest(SentenceEmbeddingsTestSpec())
suite.addTest(StopWordsCleanerTestSpec())
suite.addTest(StopWordsCleanerModelTestSpec())
suite.addTest(NGramGeneratorTestSpec())
suite.addTest(ChunkEmbeddingsTestSpec())
suite.addTest(EmbeddingsFinisherTestSpec())
suite.addTest(NerDLModelTestSpec())
suite.addTest(YakeKeywordExtractionTestSpec())
suite.addTest(SentenceDetectorDLTestSpec())
suite.addTest(SentenceDetectorDLExtraParamsTestSpec())
suite.addTest(WordSegmenterTestSpec())
suite.addTest(LanguageDetectorDLTestSpec())
# suite.addTest(GraphExtractionTestSpec())  # TODO
suite.addTest(EntityRulerTestSpec())
suite.addTest(Doc2VecTestSpec())
suite.addTest(AlbertForTokenClassificationTestSpec())
suite.addTest(Word2VecTestSpec())

# Should be locally tested
# suite.addTest(ElmoEmbeddingsTestSpec())
# suite.addTest(AlbertEmbeddingsTestSpec())
# suite.addTest(XlnetEmbeddingsTestSpec())
# suite.addTest(UniversalSentenceEncoderTestSpec())
# suite.addTest(ClassifierDLTestSpec())
# suite.addTest(MultiClassifierDLTestSpec())
# suite.addTest(SentimentDLTestSpec())
# suite.addTest(RecursiveTestSpec())  # TODO
# suite.addTest(T5TransformerQATestSpec())
# suite.addTest(T5TransformerSummaryTestSpec())
# suite.addTest(T5TransformerSummaryWithSamplingTestSpec())
# suite.addTest(T5TransformerSummaryWithSamplingAndDeactivatedTopKTestSpec())
# suite.addTest(T5TransformerSummaryWithSamplingAndTemperatureTestSpec())
# suite.addTest(T5TransformerSummaryWithSamplingAndTopPTestSpec())
# suite.addTest(T5TransformerSummaryWithRepetitionPenaltyTestSpec())
# suite.addTest(BertForTokenClassificationTestSpec())
# suite.addTest(RoBertaSentenceEmbeddingsTestSpec())
# suite.addTest(RoBertaForTokenClassificationTestSpec())
# suite.addTest(XlmRoBertaForTokenClassificationTestSpec())
# suite.addTest(XlnetForTokenClassificationTestSpec())
# suite.addTest(LongformerForTokenClassificationTestSpec())
# suite.addTest(DistilBertForSequenceClassificationTestSpec())
# suite.addTest(RoBertaForSequenceClassificationTestSpec())
# suite.addTest(XlmRoBertaForSequenceClassificationTestSpec())
# suite.addTest(GetClassesTestSpec())
# suite.addTest(GPT2TransformerTextGenerationTestSpec())
# suite.addTest(DeBertaForSequenceClassificationTestSpec())  # TODO
# suite.addTest(DeBertaForTokenClassificationTestSpec())  # TODO
# suite.addTest(CamemBertEmbeddingsTestSpec())  # TODO
#
# Misc tests
suite.addTest(UtilitiesTestSpec())
suite.addTest(SerializersTestSpec())
suite.addTest(ResourceDownloaderShowTestSpec())

# Functions tests
suite.addTest(FunctionMapColumnsTestSpec())
suite.addTest(FunctionMapColumnTestSpec())


if __name__ == '__main__':

    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
