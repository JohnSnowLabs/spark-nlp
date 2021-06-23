from test.annotators import *
from test.functions import FunctionMapColumnsTestSpec, FunctionMapColumnTestSpec
from test.misc import *
from test.base import *

# Annotator tests
unittest.TextTestRunner().run(BasicAnnotatorsTestSpec())
unittest.TextTestRunner().run(RegexMatcherTestSpec())
unittest.TextTestRunner().run(TokenizerTestSpec())
unittest.TextTestRunner().run(NormalizerTestSpec())
unittest.TextTestRunner().run(ChunkTokenizerTestSpec())
unittest.TextTestRunner().run(LemmatizerTestSpec())
unittest.TextTestRunner().run(LemmatizerWithTrainingDataSetTestSpec())
unittest.TextTestRunner().run(DateMatcherTestSpec())
unittest.TextTestRunner().run(TextMatcherTestSpec())
unittest.TextTestRunner().run(DocumentNormalizerSpec())

unittest.TextTestRunner().run(PerceptronApproachTestSpec())
unittest.TextTestRunner().run(ChunkerTestSpec())
unittest.TextTestRunner().run(ChunkDocSerializingTestSpec())

unittest.TextTestRunner().run(PragmaticSBDTestSpec())
unittest.TextTestRunner().run(PragmaticScorerTestSpec())
unittest.TextTestRunner().run(PipelineTestSpec())
unittest.TextTestRunner().run(SpellCheckerTestSpec())
unittest.TextTestRunner().run(SymmetricDeleteTestSpec())
unittest.TextTestRunner().run(ContextSpellCheckerTestSpec())
unittest.TextTestRunner().run(ParamsGettersTestSpec())
unittest.TextTestRunner().run(DependencyParserTreeBankTestSpec())
unittest.TextTestRunner().run(DependencyParserConllUTestSpec())
unittest.TextTestRunner().run(TypedDependencyParserConll2009TestSpec())
unittest.TextTestRunner().run(TypedDependencyParserConllUTestSpec())
unittest.TextTestRunner().run(SentenceEmbeddingsTestSpec())
unittest.TextTestRunner().run(StopWordsCleanerTestSpec())
unittest.TextTestRunner().run(StopWordsCleanerModelTestSpec())
unittest.TextTestRunner().run(NGramGeneratorTestSpec())
unittest.TextTestRunner().run(ChunkEmbeddingsTestSpec())
unittest.TextTestRunner().run(EmbeddingsFinisherTestSpec())
unittest.TextTestRunner().run(NerDLModelTestSpec())
unittest.TextTestRunner().run(YakeModelTestSpec())
unittest.TextTestRunner().run(SentenceDetectorDLTestSpec())
unittest.TextTestRunner().run(WordSegmenterTestSpec())
unittest.TextTestRunner().run(LanguageDetectorDLTestSpec())

# Should be locally tested
# unittest.TextTestRunner().run(ElmoEmbeddingsTestSpec())
# unittest.TextTestRunner().run(AlbertEmbeddingsTestSpec())
# unittest.TextTestRunner().run(XlnetEmbeddingsTestSpec())
# unittest.TextTestRunner().run(UniversalSentenceEncoderTestSpec())
# unittest.TextTestRunner().run(ClassifierDLTestSpec())
# unittest.TextTestRunner().run(MultiClassifierDLTestSpec())
# unittest.TextTestRunner().run(SentimentDLTestSpec())
# unittest.TextTestRunner().run(RecursiveTestSpec())
# unittest.TextTestRunner().run(T5TransformerQATestSpec())
# unittest.TextTestRunner().run(T5TransformerSummaryTestSpec())
# unittest.TextTestRunner().run(T5TransformerSummaryWithSamplingTestSpec())
# unittest.TextTestRunner().run(T5TransformerSummaryWithSamplingAndDeactivatedTopKTestSpec())
# unittest.TextTestRunner().run(T5TransformerSummaryWithSamplingAndTemperatureTestSpec())
# unittest.TextTestRunner().run(T5TransformerSummaryWithSamplingAndTopPTestSpec())
# unittest.TextTestRunner().run(T5TransformerSummaryWithRepetitionPenaltyTestSpec())
# unittest.TextTestRunner().run(GraphExtractorTestSpec())

# Misc tests

unittest.TextTestRunner().run(UtilitiesTestSpec())
unittest.TextTestRunner().run(SerializersTestSpec())

#Functions tests
unittest.TextTestRunner().run(FunctionMapColumnsTestSpec())
unittest.TextTestRunner().run(FunctionMapColumnTestSpec())

