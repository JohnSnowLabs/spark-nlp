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

from sparknlp.common import *


class BertForQuestionAnswering(AnnotatorModel,
                               HasCaseSensitiveProperties,
                               HasBatchedAnnotate,
                               HasEngine,
                               HasMaxSentenceLengthLimit):
    """BertForQuestionAnswering can load BERT Models with a span classification head on top for extractive
    question-answering tasks like SQuAD (a linear layer on top of the hidden-states output to compute span start
    logits and span end logits).

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> spanClassifier = BertForQuestionAnswering.pretrained() \\
    ...     .setInputCols(["document_question", "document_context"]) \\
    ...     .setOutputCol("answer")

    The default model is ``"bert_base_cased_qa_squad2"``, if no name is
    provided.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Question+Answering>`__.

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
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    maxSentenceLength
        Max sentence length to process, by default 128

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = MultiDocumentAssembler() \\
    ...     .setInputCols(["question", "context"]) \\
    ...     .setOutputCols(["document_question", "document_context"])
    >>> questionAnswering = BertForQuestionAnswering.pretrained() \\
    ...     .setInputCols(["document_question", "document_context"]) \\
    ...     .setOutputCol("answer") \\
    ...     .setCaseSensitive(False)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     questionAnswering
    ... ])
    >>> data = spark.createDataFrame([["What's my name?", "My name is Clara and I live in Berkeley."]]).toDF("question", "context")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("answer.result").show(truncate=False)
    +--------------------+
    |result              |
    +--------------------+
    |[Clara]             |
    +--------------------+
    """
    name = "BertForQuestionAnswering"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.CHUNK

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    coalesceSentences = Param(Params._dummy(), "coalesceSentences",
                              "Instead of 1 class per sentence (if inputCols is '''sentence''') output 1 class per document by averaging probabilities in all sentences.",
                              TypeConverters.toBoolean)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.BertForQuestionAnswering",
                 java_model=None):
        super(BertForQuestionAnswering, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            maxSentenceLength=128,
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
        BertForQuestionAnswering
            The restored model
        """
        from sparknlp.internal import _BertQuestionAnsweringLoader
        jModel = _BertQuestionAnsweringLoader(folder, spark_session._jsparkSession)._java_obj
        return BertForQuestionAnswering(java_model=jModel)

    @staticmethod
    def pretrained(name="bert_base_cased_qa_squad2", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "bert_base_cased_qa_squad2"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        BertForQuestionAnswering
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BertForQuestionAnswering, name, lang, remote_loc)
