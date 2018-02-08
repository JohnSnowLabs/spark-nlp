from test.annotators import *
from test.misc import *

# Annotator tests

unittest.TextTestRunner().run(BasicAnnotatorsTestSpec())
unittest.TextTestRunner().run(RegexMatcherTestSpec())
unittest.TextTestRunner().run(LemmatizerTestSpec())
unittest.TextTestRunner().run(DateMatcherTestSpec())
unittest.TextTestRunner().run(EntityExtractorTestSpec())
unittest.TextTestRunner().run(PerceptronApproachTestSpec())
unittest.TextTestRunner().run(PragmaticSBDTestSpec())
unittest.TextTestRunner().run(PragmaticScorerTestSpec())
unittest.TextTestRunner().run(PipelineTestSpec())
unittest.TextTestRunner().run(SpellCheckerTestSpec())

# Misc tests
unittest.TextTestRunner().run(UtilitiesTestSpec())
unittest.TextTestRunner().run(ConfigPathTestSpec())
