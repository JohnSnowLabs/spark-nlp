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
# New Annotators need to be imported here
from sparknlp.annotator.classifier_dl import *
from sparknlp.annotator.embeddings import *
from sparknlp.annotator.er import *
from sparknlp.annotator.keyword_extraction import *
from sparknlp.annotator.ld_dl import *
from sparknlp.annotator.matcher import *
from sparknlp.annotator.ner import *
from sparknlp.annotator.dependency import *
from sparknlp.annotator.pos import *
from sparknlp.annotator.sentence import *
from sparknlp.annotator.sentiment import *
from sparknlp.annotator.seq2seq import *
from sparknlp.annotator.spell_check import *
from sparknlp.annotator.token import *
from sparknlp.annotator.ws import *
from sparknlp.annotator.chunker import *
from sparknlp.annotator.document_normalizer import *
from sparknlp.annotator.graph_extraction import *
from sparknlp.annotator.lemmatizer import *
from sparknlp.annotator.n_gram_generator import *
from sparknlp.annotator.normalizer import *
from sparknlp.annotator.stemmer import *
from sparknlp.annotator.stop_words_cleaner import *
from sparknlp.annotator.coref import *
from sparknlp.annotator.tf_ner_dl_graph_builder import *
from sparknlp.annotator.cv import *
from sparknlp.annotator.audio import *
from sparknlp.annotator.chunk2_doc import *
from sparknlp.annotator.date2_chunk import *
from sparknlp.annotator.openai import *
from sparknlp.annotator.token2_chunk import *
from sparknlp.annotator.document_character_text_splitter import *
from sparknlp.annotator.document_token_splitter import *

if sys.version_info[0] == 2:
    raise ImportError(
        "Spark NLP only supports Python 3.6 and above. "
        "Please use Python 3.6 or above that is compatible with both Spark NLP and PySpark"
    )
else:
    __import__("com.johnsnowlabs.nlp")

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
coref = sys.modules[__name__]
cv = sys.modules[__name__]
audio = sys.modules[__name__]
