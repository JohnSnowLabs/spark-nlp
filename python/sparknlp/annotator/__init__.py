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

"""Module containing all available Annotators of Spark NLP and their base
classes.
"""

import sys

from sparknlp.common import *

# Do NOT delete. Looks redundant but this is key work around for python 2 support.
if sys.version_info[0] == 2:
    from sparknlp.base import DocumentAssembler, Finisher, EmbeddingsFinisher, TokenAssembler
else:
    import com.johnsnowlabs.nlp

annotators = sys.modules[__name__]
pos = sys.modules[__name__]
pos.perceptron = sys.modules[__name__]
ner = sys.modules[__name__]
ner.crf = sys.modules[__name__]
ner.dl = sys.modules[__name__]
regex = sys.modules[__name__]
sbd = sys.modules[__name__]
sbd.pragmatic = sys.modules[__name__]
sda = sys.modules[__name__]
sda.pragmatic = sys.modules[__name__]
sda.vivekn = sys.modules[__name__]
spell = sys.modules[__name__]
spell.norvig = sys.modules[__name__]
spell.symmetric = sys.modules[__name__]
spell.context = sys.modules[__name__]
parser = sys.modules[__name__]
parser.dep = sys.modules[__name__]
parser.typdep = sys.modules[__name__]
embeddings = sys.modules[__name__]
classifier = sys.modules[__name__]
classifier.dl = sys.modules[__name__]
ld = sys.modules[__name__]
ld.dl = sys.modules[__name__]
keyword = sys.modules[__name__]
keyword.yake = sys.modules[__name__]
sentence_detector_dl = sys.modules[__name__]
seq2seq = sys.modules[__name__]
ws = sys.modules[__name__]
er = sys.modules[__name__]
