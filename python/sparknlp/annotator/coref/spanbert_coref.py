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
"""Contains classes for the SpanBertCorefModel."""

from sparknlp.common import *


class SpanBertCorefModel(AnnotatorModel,
                         HasEmbeddingsProperties,
                         HasCaseSensitiveProperties,
                         HasStorageRef,
                         HasEngine,
                         HasMaxSentenceLengthLimit):
    """
    A coreference resolution model based on SpanBert.

    A coreference resolution model identifies expressions which refer to the same entity in a text. For example, given
    a sentence "John told Mary he would like to borrow a book from her." the model will link "he" to "John" and "her"
    to "Mary".

    This model is based on SpanBert, which is fine-tuned on the OntoNotes 5.0 data set.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object:

    >>> corefResolution = SpanBertCorefModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("coref")

    The default model is ``"spanbert_base_coref"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?q=coref>`__.

    For extended examples of usage, see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/coreference-resolution/Coreference_Resolution_SpanBertCorefModel.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``DEPENDENCY``
    ====================== ======================

    Parameters
    ----------
    maxSentenceLength
        Maximum sentence length to process
    maxSegmentLength
        Maximum segment length
    textGenre
        Text genre. One of the following values:

        | "bc", // Broadcast conversation, default
        | "bn", // Broadcast news
        | "nw", // News wire
        | "pt", // Pivot text: Old Testament and New Testament text
        | "tc", // Telephone conversation
        | "wb" // Web data

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> corefResolution = SpanBertCorefModel() \\
    ...     .pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("corefs") \\
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     corefResolution
    ... ])
    >>> data = spark.createDataFrame([
    ...     ["John told Mary he would like to borrow a book from her."]
    ... ]).toDF("text")
    >>> results = pipeline.fit(data).transform(data))
    >>> results \\
    ...     .selectExpr("explode(corefs) AS coref")
    ...     .selectExpr("coref.result as token", "coref.metadata")
    ...     .show(truncate=False)
    +-----+------------------------------------------------------------------------------------+
    |token|metadata                                                                            |
    +-----+------------------------------------------------------------------------------------+
    |John |{head.sentence -> -1, head -> ROOT, head.begin -> -1, head.end -> -1, sentence -> 0}|
    |he   |{head.sentence -> 0, head -> John, head.begin -> 0, head.end -> 3, sentence -> 0}   |
    |Mary |{head.sentence -> -1, head -> ROOT, head.begin -> -1, head.end -> -1, sentence -> 0}|
    |her  |{head.sentence -> 0, head -> Mary, head.begin -> 10, head.end -> 13, sentence -> 0} |
    +-----+------------------------------------------------------------------------------------|
    """

    name = "SpanBertCorefModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.DEPENDENCY

    maxSegmentLength = Param(Params._dummy(),
                             "maxSegmentLength",
                             "Max segment length",
                             typeConverter=TypeConverters.toInt)

    textGenre = Param(Params._dummy(),
                      "textGenre",
                      "Text genre, one of ('bc', 'bn', 'mz', 'nw', 'pt','tc', 'wb')",
                      typeConverter=TypeConverters.toString)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSegmentLength(self, value):
        """Sets max segment length

        Parameters
        ----------
        value : int
            Max segment length
        """
        return self._set(maxSegmentLength=value)

    def setTextGenre(self, value):
        """ Sets the text genre, one of the following values:
            | "bc" : Broadcast conversation, default
            | "bn"  Broadcast news
            | "nw" : News wire
            | "pt" : Pivot text: Old Testament and New Testament text
            | "tc" : Telephone conversation
            | "wb" : Web data

        Parameters
        ----------
        value : string
            Text genre code, default is 'bc'
        """
        return self._set(textGenre=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.coref.SpanBertCorefModel", java_model=None):
        super(SpanBertCorefModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            maxSentenceLength=512,
            caseSensitive=True,
            textGenre="bc"
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
        SpanBertCorefModel
            The restored model
        """
        from sparknlp.internal import _SpanBertCorefLoader
        jModel = _SpanBertCorefLoader(folder, spark_session._jsparkSession)._java_obj
        return SpanBertCorefModel(java_model=jModel)

    @staticmethod
    def pretrained(name="spanbert_base_coref", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "spanbert_base_coref"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        SpanBertCorefModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SpanBertCorefModel, name, lang, remote_loc)
