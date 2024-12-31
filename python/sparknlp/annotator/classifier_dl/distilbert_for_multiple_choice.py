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

from sparknlp.common import *

class DistilBertForMultipleChoice(AnnotatorModel,
                                HasCaseSensitiveProperties,
                                HasBatchedAnnotate,
                                HasEngine,
                                HasMaxSentenceLengthLimit):
    """DistilBertForMultipleChoice can load DistilBert Models with a multiple choice classification head on top
    (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> spanClassifier = DistilBertForMultipleChoice.pretrained() \\
    ...     .setInputCols(["document_question", "document_context"]) \\
    ...     .setOutputCol("answer")

    The default model is ``"bert_base_uncased_multiple_choice"``, if no name is
    provided.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Multiple+Choice>`__.

    To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, DOCUMENT``    ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 8
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        False
    maxSentenceLength
        Max sentence length to process, by default 512

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = MultiDocumentAssembler() \\
    ...     .setInputCols(["question", "context"]) \\
    ...     .setOutputCols(["document_question", "document_context"])
    >>> questionAnswering = DistilBertForMultipleChoice.pretrained() \\
    ...     .setInputCols(["document_question", "document_context"]) \\
    ...     .setOutputCol("answer") \\
    ...     .setCaseSensitive(False)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     questionAnswering
    ... ])
    >>> data = spark.createDataFrame([["The Eiffel Tower is located in which country??", "Germany, France, Italy"]]).toDF("question", "context")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("answer.result").show(truncate=False)
    +--------------------+
    |result              |
    +--------------------+
    |[France]             |
    +--------------------+
    """
    name = "DistilBertForMultipleChoice"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.CHUNK

    choicesDelimiter = Param(Params._dummy(),
                             "choicesDelimiter",
                             "Delimiter character use to split the choices",
                             TypeConverters.toString)

    def setChoicesDelimiter(self, value):
        """Sets delimiter character use to split the choices

        Parameters
        ----------
        value : string
            Delimiter character use to split the choices
        """
        return self._set(caseSensitive=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForMultipleChoice",
                 java_model=None):
        super(DistilBertForMultipleChoice, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=4,
            maxSentenceLength=512,
            caseSensitive=False,
            choicesDelimiter = ","
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
        DistilBertForMultipleChoice
            The restored model
        """
        from sparknlp.internal import _DistilBertMultipleChoiceLoader
        jModel = _DistilBertMultipleChoiceLoader(folder, spark_session._jsparkSession)._java_obj
        return DistilBertForMultipleChoice(java_model=jModel)

    @staticmethod
    def pretrained(name="distilbert_base_uncased_multiple_choice", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "bert_base_uncased_multiple_choice"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        DistilBertForMultipleChoice
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(DistilBertForMultipleChoice, name, lang, remote_loc)