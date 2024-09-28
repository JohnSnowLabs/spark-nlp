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
"""Contains classes for CamemBertForSequenceClassification."""

from sparknlp.common import *


class CamemBertForZeroShotClassification(AnnotatorModel,
                                         HasCaseSensitiveProperties,
                                         HasBatchedAnnotate,
                                         HasClassifierActivationProperties,
                                         HasCandidateLabelsProperties,
                                         HasEngine,
                                         HasMaxSentenceLengthLimit):
    """CamemBertForZeroShotClassification using a `ModelForSequenceClassification` trained on NLI (natural language
    inference) tasks. Equivalent of `DeBertaForSequenceClassification` models, but these models don't require a hardcoded
    number of potential classes, they can be chosen at runtime. It usually means it's slower but it is much more
    flexible.
    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pretrained model.
    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:
    >>> sequenceClassifier = CamemBertForZeroShotClassification.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("label")
    The default model is ``"camembert_zero_shot_classifier_xnli_onnx"``, if no name is
    provided.
    For available pretrained models please see the `Models Hub
    <https://sparknlp.orgtask=Text+Classification>`__.
    To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.
    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CATEGORY``
    ====================== ======================
    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 8
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        True
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    maxSentenceLength
        Max sentence length to process, by default 128
    coalesceSentences
        Instead of 1 class per sentence (if inputCols is `sentence`) output 1
        class per document by averaging probabilities in all sentences, by
        default False
    activation
        Whether to calculate logits via Softmax or Sigmoid, by default
        `"softmax"`.
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
    >>> sequenceClassifier = CamemBertForZeroShotClassification.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("multi_class") \\
    ...     .setCaseSensitive(True)
    ...     .setCandidateLabels(["sport", "politique", "science"])
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     sequenceClassifier
    ... ])
    >>> data = spark.createDataFrame([["L'équipe de France joue aujourd'hui au Parc des Princes"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("class.result").show(truncate=False)
    +------+
    |result|
    +------+
    |[sport]|
    +------+
    """
    name = "CamemBertForZeroShotClassification"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CATEGORY

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    coalesceSentences = Param(Params._dummy(), "coalesceSentences",
                              "Instead of 1 class per sentence (if inputCols is '''sentence''') output 1 class per document by averaging probabilities in all sentences.",
                              TypeConverters.toBoolean)

    def getClasses(self):
        """
        Returns labels used to train this model
        """
        return self._call_java("getClasses")

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setCoalesceSentences(self, value):
        """Instead of 1 class per sentence (if inputCols is '''sentence''') output 1
        class per document by averaging probabilities in all sentences, by default True.

        Due to max sequence length limit in almost all transformer models such as BERT
        (512 tokens), this parameter helps feeding all the sentences into the model and
        averaging all the probabilities for the entire document instead of probabilities
        per sentence.

        Parameters
        ----------
        value : bool
            If the output of all sentences will be averaged to one output
        """
        return self._set(coalesceSentences=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.CamemBertForZeroShotClassification",
                 java_model=None):
        super(CamemBertForZeroShotClassification, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=True,
            coalesceSentences=False,
            activation="softmax"
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
        CamemBertForZeroShotClassification
            The restored model
        """
        from sparknlp.internal import _CamemBertForZeroShotClassificationLoader
        jModel = _CamemBertForZeroShotClassificationLoader(folder, spark_session._jsparkSession)._java_obj
        return CamemBertForZeroShotClassification(java_model=jModel)

    @staticmethod
    def pretrained(name="camembert_zero_shot_classifier_xnli_onnx", lang="fr", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "camembert_zero_shot_classifier_xnli_onnx"
        lang : str, optional
            Language of the pretrained model, by default "fr"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        CamemBertForSequenceClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(CamemBertForZeroShotClassification, name, lang, remote_loc)
