from test.annotators import *
from test.misc import *

# Annotator tests
unittest.TextTestRunner().run(BasicAnnotatorsTestSpec())
unittest.TextTestRunner().run(RegexMatcherTestSpec())
unittest.TextTestRunner().run(TokenizerTestSpec())
unittest.TextTestRunner().run(LemmatizerTestSpec())
unittest.TextTestRunner().run(DateMatcherTestSpec())
unittest.TextTestRunner().run(TextMatcherTestSpec())

unittest.TextTestRunner().run(PerceptronApproachTestSpec())
unittest.TextTestRunner().run(ChunkerTestSpec())

unittest.TextTestRunner().run(PragmaticSBDTestSpec())
unittest.TextTestRunner().run(PragmaticScorerTestSpec())
unittest.TextTestRunner().run(PipelineTestSpec())
unittest.TextTestRunner().run(SpellCheckerTestSpec())
unittest.TextTestRunner().run(SymmetricDeleteTestSpec())
unittest.TextTestRunner().run(ParamsGettersTestSpec())

# Misc tests
unittest.TextTestRunner().run(UtilitiesTestSpec())
unittest.TextTestRunner().run(ConfigPathTestSpec())
unittest.TextTestRunner().run(SerializersTestSpec())
unittest.TextTestRunner().run(OcrTestSpec())
