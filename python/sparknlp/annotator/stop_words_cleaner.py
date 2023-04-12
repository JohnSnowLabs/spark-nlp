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
"""Contains classes for the StopWordsCleaner."""
from sparknlp.common import *


class StopWordsCleaner(AnnotatorModel):
    """This annotator takes a sequence of strings (e.g. the output of a
    Tokenizer, Normalizer, Lemmatizer, and Stemmer) and drops all the stop words
    from the input sequences.

    By default, it uses stop words from MLlibs `StopWordsRemover
    <https://spark.apache.org/docs/latest/ml-features#stopwordsremover>`__. Stop
    words can also be defined by explicitly setting them with
    :meth:`.setStopWords` or loaded from pretrained models using ``pretrained``
    of its companion object.


    >>> stopWords = StopWordsCleaner.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("cleanTokens")

    This will load the default pretrained model ``"stopwords_en"``.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.orgtask=Stop+Words+Removal>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/stop-words/StopWordsCleaner.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    stopWords
        The words to be filtered out, by default english stopwords from Spark ML
    caseSensitive
        Whether to consider case, by default False
    locale
        Locale of the input. ignored when case sensitive, by default locale of
        the JVM

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
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
    >>> stopWords = StopWordsCleaner() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("cleanTokens") \\
    ...     .setCaseSensitive(False)
    >>> pipeline = Pipeline().setStages([
    ...       documentAssembler,
    ...       sentenceDetector,
    ...       tokenizer,
    ...       stopWords
    ...     ])
    >>> data = spark.createDataFrame([
    ...     ["This is my first sentence. This is my second."],
    ...     ["This is my third sentence. This is my forth."]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("cleanTokens.result").show(truncate=False)
    +-------------------------------+
    |result                         |
    +-------------------------------+
    |[first, sentence, ., second, .]|
    |[third, sentence, ., forth, .] |
    +-------------------------------+
    """

    name = "StopWordsCleaner"

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.StopWordsCleaner", java_model=None):
        super(StopWordsCleaner, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            stopWords=StopWordsCleaner.loadDefaultStopWords("english"),
            caseSensitive=False,
            locale=self._java_obj.getLocale()
        )

    stopWords = Param(Params._dummy(), "stopWords", "The words to be filtered out",
                      typeConverter=TypeConverters.toListString)
    caseSensitive = Param(Params._dummy(), "caseSensitive", "whether to do a case sensitive " +
                          "comparison over the stop words", typeConverter=TypeConverters.toBoolean)
    locale = Param(Params._dummy(), "locale", "locale of the input. ignored when case sensitive " +
                   "is true", typeConverter=TypeConverters.toString)

    def setStopWords(self, value):
        """Sets the words to be filtered out, by default english stopwords from
        Spark ML.

        Parameters
        ----------
        value : List[str]
            The words to be filtered out
        """
        return self._set(stopWords=value)

    def setCaseSensitive(self, value):
        """Sets whether to do a case sensitive comparison, by default False.

        Parameters
        ----------
        value : bool
            Whether to do a case sensitive comparison
        """
        return self._set(caseSensitive=value)

    def setLocale(self, value):
        """Sets locale of the input. Ignored when case sensitive, by default
        locale of the JVM.

        Parameters
        ----------
        value : str
            Locale of the input
        """
        return self._set(locale=value)

    def loadDefaultStopWords(language="english"):
        """Loads the default stop words for the given language.

        Supported languages: danish, dutch, english, finnish, french, german,
        hungarian, italian, norwegian, portuguese, russian, spanish, swedish,
        turkish

        Parameters
        ----------
        language : str, optional
            Language stopwords to load, by default "english"
        """
        from pyspark.ml.wrapper import _jvm
        stopWordsObj = _jvm().org.apache.spark.ml.feature.StopWordsRemover
        return list(stopWordsObj.loadDefaultStopWords(language))

    @staticmethod
    def pretrained(name="stopwords_en", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "stopwords_en"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        StopWordsCleaner
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(StopWordsCleaner, name, lang, remote_loc)
