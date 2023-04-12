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
"""Contains classes for the WordSegmenter."""

from sparknlp.common import *


class WordSegmenterApproach(AnnotatorApproach):
    """Trains a WordSegmenter which tokenizes non-english or non-whitespace
    separated texts.

    Many languages are not whitespace separated and their sentences are a concatenation
    of many symbols, like Korean, Japanese or Chinese. Without understanding the
    language, splitting the words into their corresponding tokens is impossible. The
    WordSegmenter is trained to understand these languages and split them into
    semantically correct parts.

    This annotator is based on the paper Chinese Word Segmentation as Character Tagging
    [1]. Word segmentation is treated as a tagging problem. Each character is be tagged
    as on of four different labels: LL (left boundary), RR (right boundary), MM (middle)
    and LR (word by itself). The label depends on the position of the word in the
    sentence. LL tagged words will combine with the word on the right. Likewise, RR
    tagged words combine with words on the left. MM tagged words are treated as the
    middle of the word and combine with either side. LR tagged words are words by
    themselves.

    Example (from [1], Example 3(a) (raw), 3(b) (tagged), 3(c) (translation)):
        - 上海 计划 到 本 世纪 末 实现 人均 国内 生产 总值 五千 美元
        - 上/LL 海/RR 计/LL 划/RR 到/LR 本/LR 世/LL 纪/RR 末/LR 实/LL 现/RR 人/LL 均/RR
          国/LL 内/RR 生/LL 产/RR 总/LL值/RR 五/LL 千/RR 美/LL 元/RR
        - Shanghai plans to reach the goal of 5,000 dollars in per capita GDP by the end
          of the century.

    For instantiated/pretrained models, see :class:`.WordSegmenterModel`.

    To train your own model, a training dataset consisting of `Part-Of-Speech
    tags <https://en.wikipedia.org/wiki/Part-of-speech_tagging>`__ is required.
    The data has to be loaded into a dataframe, where the column is an
    Annotation of type ``POS``. This can be set with :meth:`.setPosColumn`.

    **Tip**:
    The helper class :class:`.POS` might be useful to read training data into
    data frames.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/chinese/word_segmentation>`__.

    References
    ----------

    `[1] <https://aclanthology.org/O03-4002.pdf>`__ Xue, Nianwen. “Chinese Word
    Segmentation as Character Tagging.” International Journal of Computational
    Linguistics & Chinese Language Processing, Volume 8, Number 1, February 2003:
    Special Issue on Word Formation and Chinese Language Processing, 2003, pp. 29-48.
    ACLWeb, https://aclanthology.org/O03-4002.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    posCol
        column of Array of POS tags that match tokens
    nIterations
        Number of iterations in training, converges to better accuracy, by
        default 5
    frequencyThreshold
        How many times at least a tag on a word to be marked as frequent, by
        default 5
    ambiguityThreshold
        How much percentage of total amount of words are covered to be marked as
        frequent, by default 0.97
    enableRegexTokenizer
        Whether to use RegexTokenizer before segmentation. Useful for multilingual text
    toLowercase
        Indicates whether to convert all characters to lowercase before tokenizing. Used only when enableRegexTokenizer is true
    pattern
        regex pattern used for tokenizing. Used only when enableRegexTokenizer is true

    Examples
    --------
    In this example, ``"chinese_train.utf8"`` is in the form of::

        十|LL 四|RR 不|LL 是|RR 四|LL 十|RR

    and is loaded with the `POS` class to create a dataframe of ``POS`` type
    Annotations.

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> wordSegmenter = WordSegmenterApproach() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token") \\
    ...     .setPosColumn("tags") \\
    ...     .setNIterations(5)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     wordSegmenter
    ... ])
    >>> trainingDataSet = POS().readDataset(
    ...     spark,
    ...     "src/test/resources/word-segmenter/chinese_train.utf8"
    ... )
    >>> pipelineModel = pipeline.fit(trainingDataSet)
    """
    name = "WordSegmenterApproach"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.TOKEN

    posCol = Param(Params._dummy(),
                   "posCol",
                   "column of Array of POS tags that match tokens",
                   typeConverter=TypeConverters.toString)

    nIterations = Param(Params._dummy(),
                        "nIterations",
                        "Number of iterations in training, converges to better accuracy",
                        typeConverter=TypeConverters.toInt)

    frequencyThreshold = Param(Params._dummy(),
                               "frequencyThreshold",
                               "How many times at least a tag on a word to be marked as frequent",
                               typeConverter=TypeConverters.toInt)

    ambiguityThreshold = Param(Params._dummy(),
                               "ambiguityThreshold",
                               "How much percentage of total amount of words are covered to be marked as frequent",
                               typeConverter=TypeConverters.toFloat)

    enableRegexTokenizer = Param(Params._dummy(),
                                 "enableRegexTokenizer",
                                 "Whether to use RegexTokenizer before segmentation. Useful for multilingual text",
                                 typeConverter=TypeConverters.toBoolean)

    toLowercase = Param(Params._dummy(),
                        "toLowercase",
                        "Indicates whether to convert all characters to lowercase before tokenizing.",
                        typeConverter=TypeConverters.toBoolean)

    pattern = Param(Params._dummy(),
                    "pattern",
                    "regex pattern used for tokenizing. Defaults \\s+",
                    typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(WordSegmenterApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.ws.WordSegmenterApproach")
        self._setDefault(
            nIterations=5, frequencyThreshold=5, ambiguityThreshold=0.97,
            enableRegexTokenizer=False, toLowercase=False, pattern="\\s+"
        )

    def setPosColumn(self, value):
        """Sets column name for array of POS tags that match tokens.

        Parameters
        ----------
        value : str
            Name of the column
        """
        return self._set(posCol=value)

    def setNIterations(self, value):
        """Sets number of iterations in training, converges to better accuracy,
        by default 5.

        Parameters
        ----------
        value : int
            Number of iterations
        """
        return self._set(nIterations=value)

    def setFrequencyThreshold(self, value):
        """Sets how many times at least a tag on a word to be marked as
        frequent, by default 5.

        Parameters
        ----------
        value : int
            Frequency threshold to be marked as frequent
        """
        return self._set(frequencyThreshold=value)

    def setAmbiguityThreshold(self, value):
        """Sets the percentage of total amount of words are covered to be
        marked as frequent, by default 0.97.

        Parameters
        ----------
        value : float
            Percentage of total amount of words are covered to be
            marked as frequent
        """
        return self._set(ambiguityThreshold=value)

    def getNIterations(self):
        """Gets number of iterations in training, converges to better accuracy.

        Returns
        -------
        int
            Number of iterations
        """
        return self.getOrDefault(self.nIterations)

    def getFrequencyThreshold(self):
        """Sets How many times at least a tag on a word to be marked as
        frequent.

        Returns
        -------
        int
            Frequency threshold to be marked as frequent
        """
        return self.getOrDefault(self.frequencyThreshold)

    def getAmbiguityThreshold(self):
        """Sets How much percentage of total amount of words are covered to be
        marked as frequent.

        Returns
        -------
        float
            Percentage of total amount of words are covered to be
            marked as frequent
        """
        return self.getOrDefault(self.ambiguityThreshold)

    def setEnableRegexTokenizer(self, value):
        """Sets whether to to use RegexTokenizer before segmentation.
        Useful for multilingual text

        Parameters
        ----------
        value : bool
            Whether to use RegexTokenizer before segmentation
        """
        return self._set(enableRegexTokenizer=value)

    def setToLowercase(self, value):
        """Sets whether to convert all characters to lowercase before
        tokenizing, by default False.

        Parameters
        ----------
        value : bool
            Whether to convert all characters to lowercase before tokenizing
        """
        return self._set(toLowercase=value)

    def setPattern(self, value):
        """Sets the regex pattern used for tokenizing, by default ``\\s+``.

        Parameters
        ----------
        value : str
            Regex pattern used for tokenizing
        """
        return self._set(pattern=value)

    def _create_model(self, java_model):
        return WordSegmenterModel(java_model=java_model)


class WordSegmenterModel(AnnotatorModel):
    """WordSegmenter which tokenizes non-english or non-whitespace separated
    texts.

    Many languages are not whitespace separated and their sentences are a
    concatenation of many symbols, like Korean, Japanese or Chinese. Without
    understanding the language, splitting the words into their corresponding
    tokens is impossible. The WordSegmenter is trained to understand these
    languages and plit them into semantically correct parts.

    This is the instantiated model of the :class:`.WordSegmenterApproach`. For
    training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> wordSegmenter = WordSegmenterModel.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("words_segmented")

    The default model is ``"wordseg_pku"``, default language is ``"zh"``, if no
    values are provided. For available pretrained models please see the `Models
    Hub <https://sparknlp.org/models?task=Word+Segmentation>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/jupyter/annotation/chinese/word_segmentation/words_segmenter_demo.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``TOKEN``
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
    >>> wordSegmenter = WordSegmenterModel.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     wordSegmenter
    ... ])
    >>> data = spark.createDataFrame([["然而，這樣的處理也衍生了一些問題。"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("token.result").show(truncate=False)
    +--------------------------------------------------------+
    |result                                                  |
    +--------------------------------------------------------+
    |[然而, ，, 這樣, 的, 處理, 也, 衍生, 了, 一些, 問題, 。      ]|
    +--------------------------------------------------------+
    """
    name = "WordSegmenterModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.TOKEN

    enableRegexTokenizer = Param(Params._dummy(),
                                 "enableRegexTokenizer",
                                 "Whether to use RegexTokenizer before segmentation. Useful for multilingual text",
                                 typeConverter=TypeConverters.toBoolean)

    toLowercase = Param(Params._dummy(),
                        "toLowercase",
                        "Indicates whether to convert all characters to lowercase before tokenizing.",
                        typeConverter=TypeConverters.toBoolean)

    pattern = Param(Params._dummy(),
                    "pattern",
                    "regex pattern used for tokenizing. Defaults \\s+",
                    typeConverter=TypeConverters.toString)

    def setEnableRegexTokenizer(self, value):
        """Sets whether to to use RegexTokenizer before segmentation.
        Useful for multilingual text

        Parameters
        ----------
        value : bool
            Whether to use RegexTokenizer before segmentation
        """
        return self._set(enableRegexTokenizer=value)

    def setToLowercase(self, value):
        """Sets whether to convert all characters to lowercase before
        tokenizing, by default False.

        Parameters
        ----------
        value : bool
            Whether to convert all characters to lowercase before tokenizing
        """
        return self._set(toLowercase=value)

    def setPattern(self, value):
        """Sets the regex pattern used for tokenizing, by default ``\\s+``.

        Parameters
        ----------
        value : str
            Regex pattern used for tokenizing
        """
        return self._set(pattern=value)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ws.WordSegmenterModel", java_model=None):
        super(WordSegmenterModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="wordseg_pku", lang="zh", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "wordseg_pku"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        WordSegmenterModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(WordSegmenterModel, name, lang, remote_loc)
