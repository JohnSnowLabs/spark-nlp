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
"""Contains classes for Token2Chunk."""


from sparknlp.common import *


class Token2Chunk(AnnotatorModel):
    """Converts ``TOKEN`` type Annotations to ``CHUNK`` type.

    This can be useful if a entities have been already extracted as ``TOKEN``
    and following annotators require ``CHUNK`` types.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    None

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> token2chunk = Token2Chunk() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("chunk")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     token2chunk
    ... ])
    >>> data = spark.createDataFrame([["One Two Three Four"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(chunk) as result").show(truncate=False)
    +------------------------------------------+
    |result                                    |
    +------------------------------------------+
    |[chunk, 0, 2, One, [sentence -> 0], []]   |
    |[chunk, 4, 6, Two, [sentence -> 0], []]   |
    |[chunk, 8, 12, Three, [sentence -> 0], []]|
    |[chunk, 14, 17, Four, [sentence -> 0], []]|
    +------------------------------------------+
    """
    name = "Token2Chunk"

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    def __init__(self):
        super(Token2Chunk, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Token2Chunk")

