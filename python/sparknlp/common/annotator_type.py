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


class AnnotatorType(object):
    AUDIO = "audio"
    DOCUMENT = "document"
    IMAGE = "image"
    TOKEN = "token"
    WORDPIECE = "wordpiece"
    WORD_EMBEDDINGS = "word_embeddings"
    SENTENCE_EMBEDDINGS = "sentence_embeddings"
    CATEGORY = "category"
    DATE = "date"
    ENTITY = "entity"
    SENTIMENT = "sentiment"
    POS = "pos"
    CHUNK = "chunk"
    NAMED_ENTITY = "named_entity"
    NEGEX = "negex"
    DEPENDENCY = "dependency"
    LABELED_DEPENDENCY = "labeled_dependency"
    LANGUAGE = "language"
    NODE = "node"
    TABLE = "table"
    DUMMY = "dummy"
    DOC_SIMILARITY_RANKINGS = "doc_similarity_rankings"
