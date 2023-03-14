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
"""Contains classes for the EmbeddingsFinisher."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param
from sparknlp.internal import AnnotatorTransformer


class EmbeddingsFinisher(AnnotatorTransformer):
    """Extracts embeddings from Annotations into a more easily usable form.

    This is useful for example:

    - WordEmbeddings,
    - Transformer based embeddings such as BertEmbeddings,
    - SentenceEmbeddings and
    - ChunkEmbeddings, etc.

    By using ``EmbeddingsFinisher`` you can easily transform your embeddings
    into array of floats or vectors which are compatible with Spark ML functions
    such as LDA, K-mean, Random Forest classifier or any other functions that
    require a ``featureCol``.

    For more extended examples see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/text-similarity/Spark_NLP_Spark_ML_Text_Similarity.ipynb
>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``EMBEDDINGS``         ``NONE``
    ====================== ======================

    Parameters
    ----------
    inputCols
        Names of input annotation columns containing embeddings
    outputCols
        Names of finished output columns
    cleanAnnotations
        Whether to remove all the existing annotation columns, by default False
    outputAsVector
        Whether to output the embeddings as Vectors instead of arrays,
        by default False

    Examples
    --------
    First extract embeddings.

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...    .setInputCol("text") \\
    ...    .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...    .setInputCols("document") \\
    ...    .setOutputCol("token")
    >>> normalizer = Normalizer() \\
    ...    .setInputCols("token") \\
    ...    .setOutputCol("normalized")
    >>> stopwordsCleaner = StopWordsCleaner() \\
    ...    .setInputCols("normalized") \\
    ...    .setOutputCol("cleanTokens") \\
    ...    .setCaseSensitive(False)
    >>> gloveEmbeddings = WordEmbeddingsModel.pretrained() \\
    ...    .setInputCols("document", "cleanTokens") \\
    ...    .setOutputCol("embeddings") \\
    ...    .setCaseSensitive(False)
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...    .setInputCols("embeddings") \\
    ...    .setOutputCols("finished_sentence_embeddings") \\
    ...    .setOutputAsVector(True) \\
    ...    .setCleanAnnotations(False)
    >>> data = spark.createDataFrame([["Spark NLP is an open-source text processing library."]]) \\
    ...    .toDF("text")
    >>> pipeline = Pipeline().setStages([
    ...    documentAssembler,
    ...    tokenizer,
    ...    normalizer,
    ...    stopwordsCleaner,
    ...    gloveEmbeddings,
    ...    embeddingsFinisher
    ... ]).fit(data)
    >>> result = pipeline.transform(data)

    Show results.

    >>> resultWithSize = result.selectExpr("explode(finished_sentence_embeddings) as embeddings")
    >>> resultWithSize.show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                      embeddings|
    +--------------------------------------------------------------------------------+
    |[0.1619900017976761,0.045552998781204224,-0.03229299932718277,-0.685609996318...|
    |[-0.42416998744010925,1.1378999948501587,-0.5717899799346924,-0.5078899860382...|
    |[0.08621499687433243,-0.15772999823093414,-0.06067200005054474,0.395359992980...|
    |[-0.4970499873161316,0.7164199948310852,0.40119001269340515,-0.05761000141501...|
    |[-0.08170200139284134,0.7159299850463867,-0.20677000284194946,0.0295659992843...|
    +--------------------------------------------------------------------------------+

    See Also
    --------
    EmbeddingsFinisher : for finishing embeddings
    """

    inputCols = Param(Params._dummy(), "inputCols", "name of input annotation cols containing embeddings", typeConverter=TypeConverters.toListString)
    outputCols = Param(Params._dummy(), "outputCols", "output EmbeddingsFinisher ouput cols", typeConverter=TypeConverters.toListString)
    cleanAnnotations = Param(Params._dummy(), "cleanAnnotations", "whether to remove all the existing annotation columns", typeConverter=TypeConverters.toBoolean)
    outputAsVector = Param(Params._dummy(), "outputAsVector", "if enabled it will output the embeddings as Vectors instead of arrays", typeConverter=TypeConverters.toBoolean)

    name = "EmbeddingsFinisher"

    @keyword_only
    def __init__(self):
        super(EmbeddingsFinisher, self).__init__(classname="com.johnsnowlabs.nlp.EmbeddingsFinisher")
        self._setDefault(
            cleanAnnotations=False,
            outputAsVector=False,
            outputCols=[]
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, *value):
        """Sets name of input annotation columns containing embeddings.

        Parameters
        ----------
        *value : str
            Input columns for the annotator
        """

        if len(value) == 1 and type(value[0]) == list:
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def setOutputCols(self, *value):
        """Sets names of finished output columns.

        Parameters
        ----------
        *value : List[str]
            Input columns for the annotator
        """

        if len(value) == 1 and type(value[0]) == list:
            return self._set(outputCols=value[0])
        else:
            return self._set(outputCols=list(value))

    def setCleanAnnotations(self, value):
        """Sets whether to remove all the existing annotation columns, by default
        False.

        Parameters
        ----------
        value : bool
            Whether to remove all the existing annotation columns
        """

        return self._set(cleanAnnotations=value)

    def setOutputAsVector(self, value):
        """Sets whether to output the embeddings as Vectors instead of arrays,
        by default False.

        Parameters
        ----------
        value : bool
            Whether to output the embeddings as Vectors instead of arrays
        """

        return self._set(outputAsVector=value)

    def getInputCols(self):
        """Gets input columns name of annotations."""
        return self.getOrDefault(self.inputCols)

    def getOutputCols(self):
        """Gets output columns name of annotations."""
        if len(self.getOrDefault(self.outputCols)) == 0:
            return ["finished_" + input_col for input_col in self.getInputCols()]
        else:
            return self.getOrDefault(self.outputCols)