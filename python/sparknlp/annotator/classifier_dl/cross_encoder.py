#  Copyright 2017-2024 John Snow Labs
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
"""Contains classes for CrossEncoder."""

from sparknlp.common import *


class CrossEncoder(AnnotatorModel,
                                            HasCaseSensitiveProperties,
                                            HasBatchedAnnotate,
                                            HasEngine):
    """CrossEncoder brings cross-encoder relevance scoring (as in
    ``sentence-transformers`` ``CrossEncoder``) into Spark NLP as a first-class annotator.

    It takes two row-aligned document columns, jointly encodes each row's pair as a single
    sequence ``[CLS] text_a [SEP] text_b [SEP]``, runs one forward pass through a BERT-family
    transformer with a single-logit regression head, and writes one score per row to a single
    output column. The logit is squashed with a sigmoid, so every score lands in ``[0, 1]``. Row
    ``i`` of the first column and row ``i`` of the second column produce row ``i`` of the output —
    there is no cross-row interaction. Any 1-query-vs-N-candidates reranking use case is a
    ``crossJoin``/``explode`` the user performs upstream.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object:

    >>> crossEncoder = CrossEncoder.pretrained() \\
    ...     .setInputCols(["document1", "document2"]) \\
    ...     .setOutputCol("score")

    The default model is ``"cross_encoder_ms_marco_minilm_l6_v2"``, if no name is provided.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Text+Classification>`__.

    To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, DOCUMENT`` ``CATEGORY``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 8
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        False

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> document = MultiDocumentAssembler() \\
    ...     .setInputCols(["query", "passage"]) \\
    ...     .setOutputCols(["document1", "document2"])
    >>> crossEncoder = CrossEncoder.pretrained() \\
    ...     .setInputCols(["document1", "document2"]) \\
    ...     .setOutputCol("score")
    >>> pipeline = Pipeline().setStages([document, crossEncoder])
    >>> data = spark.createDataFrame([
    ...     ["How many people live in Berlin?", "Berlin is well known for its museums."]
    ... ]).toDF("query", "passage")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("score.result").show(truncate=False)
    """
    name = "CrossEncoder"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.CATEGORY

    @keyword_only
    def __init__(self,
                 classname="com.johnsnowlabs.nlp.annotators.classifier.dl.CrossEncoder",
                 java_model=None):
        super(CrossEncoder, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            caseSensitive=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        CrossEncoder
            The restored model
        """
        from sparknlp.internal import _CrossEncoderLoader
        jModel = _CrossEncoderLoader(folder, spark_session._jsparkSession)._java_obj
        return CrossEncoder(java_model=jModel)

    @staticmethod
    def pretrained(name="cross_encoder_ms_marco_minilm_l6_v2", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "cross_encoder_ms_marco_minilm_l6_v2"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        CrossEncoder
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(CrossEncoder, name, lang, remote_loc)
