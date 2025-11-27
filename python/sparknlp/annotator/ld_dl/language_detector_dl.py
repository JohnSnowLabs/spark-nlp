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
"""Contains classes for LanguageDetectorDL."""

from sparknlp.common import *


class LanguageDetectorDL(AnnotatorModel, HasStorageRef, HasEngine):
    """Language Identification and Detection by using CNN and RNN architectures
    in TensorFlow.

    ``LanguageDetectorDL`` is an annotator that detects the language of
    documents or sentences depending on the inputCols. The models are trained on
    large datasets such as Wikipedia and Tatoeba. Depending on the language
    (how similar the characters are), the LanguageDetectorDL works best with
    text longer than 140 characters. The output is a language code in
    `Wiki Code style <https://en.wikipedia.org/wiki/List_of_Wikipedias>`__.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> languageDetector = LanguageDetectorDL.pretrained() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("language")

    The default model is ``"ld_wiki_tatoeba_cnn_21"``, default language is
    ``"xx"`` (meaning multi-lingual), if no values are provided.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models?task=Language+Detection>`__.

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/language-detection/Language_Detection_and_Indentification.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``LANGUAGE``
    ====================== ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    threshold
        The minimum threshold for the final result otheriwse it will be either
        neutral or the value set in thresholdLabel, by default 0.5
    thresholdLabel
        In case the score is less than threshold, what should be the label, by
        default Unknown
    coalesceSentences
        If sets to true the output of all sentences will be averaged to one
        output instead of one output per sentence, by default True.
    languages
       The languages used to trained the model

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> languageDetector = LanguageDetectorDL.pretrained() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("language")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       languageDetector
    ...     ])
    >>> data = spark.createDataFrame([
    ...     ["Spark NLP is an open-source text processing library for advanced natural language processing for the Python, Java and Scala programming languages."],
    ...     ["Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala."],
    ...     ["Spark NLP ist eine Open-Source-Textverarbeitungsbibliothek für fortgeschrittene natürliche Sprachverarbeitung für die Programmiersprachen Python, Java und Scala."]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("language.result").show(truncate=False)
    +------+
    |result|
    +------+
    |[en]  |
    |[fr]  |
    |[de]  |
    +------+
    """
    name = "LanguageDetectorDL"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.LANGUAGE

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL", java_model=None):
        super(LanguageDetectorDL, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            threshold=0.5,
            thresholdLabel="Unknown",
            coalesceSentences=True
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for the final result otheriwse it will be either neutral or the value set in thresholdLabel.",
                      TypeConverters.toFloat)

    thresholdLabel = Param(Params._dummy(), "thresholdLabel",
                           "In case the score is less than threshold, what should be the label. Default is neutral.",
                           TypeConverters.toString)

    coalesceSentences = Param(Params._dummy(), "coalesceSentences",
                              "If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to false.",
                              TypeConverters.toBoolean)

    languages = Param(Params._dummy(), "languages",
                      "get the languages used to trained the model",
                      TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setThreshold(self, v):
        """Sets the minimum threshold for the final result otherwise it will be
        either neutral or the value set in thresholdLabel, by default 0.5.

        Parameters
        ----------
        v : float
            Minimum threshold for the final result
        """
        self._set(threshold=v)
        return self

    def setThresholdLabel(self, p):
        """Sets what should be the label in case the score is less than
        threshold, by default Unknown.

        Parameters
        ----------
        p : str
            The replacement label.
        """
        return self._set(thresholdLabel=p)

    def setCoalesceSentences(self, value):
        """Sets if the output of all sentences will be averaged to one output
        instead of one output per sentence, by default True.

        Parameters
        ----------
        value : bool
            If the output of all sentences will be averaged to one output
        """
        return self._set(coalesceSentences=value)

    @staticmethod
    def pretrained(name="ld_wiki_tatoeba_cnn_21", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "ld_wiki_tatoeba_cnn_21"
        lang : str, optional
            Language of the pretrained model, by default "xx"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        LanguageDetectorDL
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(LanguageDetectorDL, name, lang, remote_loc)
