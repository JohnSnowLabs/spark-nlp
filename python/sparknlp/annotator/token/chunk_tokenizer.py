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
"""Contains classes for the ChunkTokenizer."""

from sparknlp.common import *
from sparknlp.annotator.token.tokenizer import Tokenizer, TokenizerModel


class ChunkTokenizer(Tokenizer):
    """Tokenizes and flattens extracted NER chunks.

    The ChunkTokenizer will split the extracted NER ``CHUNK`` type Annotations
    and will create ``TOKEN`` type Annotations.
    The result is then flattened, resulting in a single array.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``CHUNK``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    None

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> sparknlp.common import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentenceDetector = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> entityExtractor = TextMatcher() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setEntities("src/test/resources/entity-extractor/test-chunks.txt", ReadAs.TEXT) \\
    ...     .setOutputCol("entity")
    >>> chunkTokenizer = ChunkTokenizer() \\
    ...     .setInputCols(["entity"]) \\
    ...     .setOutputCol("chunk_token")
    >>> pipeline = Pipeline().setStages([
    ...         documentAssembler,
    ...         sentenceDetector,
    ...         tokenizer,
    ...         entityExtractor,
    ...         chunkTokenizer
    ... ])
    >>> data = spark.createDataFrame([
    ...     ["Hello world, my name is Michael, I am an artist and I work at Benezar"],
    ...     ["Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week."]
    >>> ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("entity.result as entity" , "chunk_token.result as chunk_token").show(truncate=False)
    +-----------------------------------------------+---------------------------------------------------+
    |entity                                         |chunk_token                                        |
    +-----------------------------------------------+---------------------------------------------------+
    |[world, Michael, work at Benezar]              |[world, Michael, work, at, Benezar]                |
    |[engineer from Farendell, last year, last week]|[engineer, from, Farendell, last, year, last, week]|
    +-----------------------------------------------+---------------------------------------------------+
    """
    name = 'ChunkTokenizer'

    inputAnnotatorTypes = [AnnotatorType.CHUNK]

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ChunkTokenizer")

    def _create_model(self, java_model):
        return ChunkTokenizerModel(java_model=java_model)


class ChunkTokenizerModel(TokenizerModel):
    """Instantiated model of the ChunkTokenizer.

    This is the instantiated model of the :class:`.ChunkTokenizer`.
    For training your own model, please see the documentation of that class.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``CHUNK``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    None
    """
    name = 'ChunkTokenizerModel'

    inputAnnotatorTypes = [AnnotatorType.CHUNK]

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ChunkTokenizerModel", java_model=None):
        super(TokenizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

