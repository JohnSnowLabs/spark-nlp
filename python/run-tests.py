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

from test.annotators import *
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
suite.addTest(PipelineTestSpec())
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
suite.addTest(MultiDocumentAssemblerTestSpec())

# Should be locally tested
# suite.addTest(ElmoEmbeddingsTestSpec())
# suite.addTest(AlbertEmbeddingsTestSpec())
# suite.addTest(XlnetEmbeddingsTestSpec())
# suite.addTest(UniversalSentenceEncoderTestSpec())
# suite.addTest(ClassifierDLTestSpec())
# suite.addTest(MultiClassifierDLTestSpec())
# suite.addTest(SentimentDLTestSpec())
# suite.addTest(RecursiveTestSpec())
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
# suite.addTest(DeBertaForSequenceClassificationTestSpec())
# suite.addTest(DeBertaForTokenClassificationTestSpec())
# suite.addTest(CamemBertEmbeddingsTestSpec())
# suite.addTest(QuestionAnsweringTestSpec())

# # Misc tests
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
