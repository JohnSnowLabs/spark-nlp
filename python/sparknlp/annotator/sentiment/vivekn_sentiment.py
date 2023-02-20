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
"""Contains classes for ViveknSentiment."""


from sparknlp.common import *
from sparknlp.common.annotator_type import AnnotatorType


class ViveknSentimentApproach(AnnotatorApproach):
    """Trains a sentiment analyser inspired by the algorithm by Vivek Narayanan.

    The analyzer requires sentence boundaries to give a score in context.
    Tokenization is needed to make sure tokens are within bounds. Transitivity
    requirements are also required.

    The training data needs to consist of a column for normalized text and a
    label column (either ``"positive"`` or ``"negative"``).

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``SENTIMENT``
    ====================== ======================

    Parameters
    ----------
    sentimentCol
        column with the sentiment result of every row. Must be 'positive' or 'negative'
    pruneCorpus
        Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1

    References
    ----------
    The algorithm is based on the paper `"Fast and accurate sentiment
    classification using an enhanced Naive Bayes model"
    <https://arxiv.org/abs/1305.6143>`__.

    https://github.com/vivekn/sentiment/

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> document = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> token = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> normalizer = Normalizer() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("normal")
    >>> vivekn = ViveknSentimentApproach() \\
    ...     .setInputCols(["document", "normal"]) \\
    ...     .setSentimentCol("train_sentiment") \\
    ...     .setOutputCol("result_sentiment")
    >>> finisher = Finisher() \\
    ...     .setInputCols(["result_sentiment"]) \\
    ...     .setOutputCols("final_sentiment")
    >>> pipeline = Pipeline().setStages([document, token, normalizer, vivekn, finisher])
    >>> training = spark.createDataFrame([
    ...     ("I really liked this movie!", "positive"),
    ...     ("The cast was horrible", "negative"),
    ...     ("Never going to watch this again or recommend it to anyone", "negative"),
    ...     ("It's a waste of time", "negative"),
    ...     ("I loved the protagonist", "positive"),
    ...     ("The music was really really good", "positive")
    ... ]).toDF("text", "train_sentiment")
    >>> pipelineModel = pipeline.fit(training)
    >>> data = spark.createDataFrame([
    ...     ["I recommend this movie"],
    ...     ["Dont waste your time!!!"]
    ... ]).toDF("text")
    >>> result = pipelineModel.transform(data)
    >>> result.select("final_sentiment").show(truncate=False)
    +---------------+
    |final_sentiment|
    +---------------+
    |[positive]     |
    |[negative]     |
    +---------------+
    """

    inputAnnotatorTypes = [AnnotatorType.TOKEN, AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTIMENT

    sentimentCol = Param(Params._dummy(),
                         "sentimentCol",
                         "column with the sentiment result of every row. Must be 'positive' or 'negative'",
                         typeConverter=TypeConverters.toString)

    pruneCorpus = Param(Params._dummy(),
                        "pruneCorpus",
                        "Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1",
                        typeConverter=TypeConverters.toInt)

    importantFeatureRatio = Param(Params._dummy(),
                                  "importantFeatureRatio",
                                  "proportion of feature content to be considered relevant. Defaults to 0.5",
                                  typeConverter=TypeConverters.toFloat)

    unimportantFeatureStep = Param(Params._dummy(),
                                   "unimportantFeatureStep",
                                   "proportion to lookahead in unimportant features. Defaults to 0.025",
                                   typeConverter=TypeConverters.toFloat)

    featureLimit = Param(Params._dummy(),
                         "featureLimit",
                         "content feature limit, to boost performance in very dirt text. Default disabled with -1",
                         typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(ViveknSentimentApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach")
        self._setDefault(pruneCorpus=1, importantFeatureRatio=0.5, unimportantFeatureStep=0.025, featureLimit=-1)

    def setSentimentCol(self, value):
        """Sets column with the sentiment result of every row.

        Must be either 'positive' or 'negative'.

        Parameters
        ----------
        value : str
            Name of the column
        """
        return self._set(sentimentCol=value)

    def setPruneCorpus(self, value):
        """Sets the removal of unfrequent scenarios from scope, by default 1.

        The higher the better performance.

        Parameters
        ----------
        value : int
            The frequency
        """
        return self._set(pruneCorpus=value)

    def _create_model(self, java_model):
        return ViveknSentimentModel(java_model=java_model)


class ViveknSentimentModel(AnnotatorModel):
    """Sentiment analyser inspired by the algorithm by Vivek Narayanan.

    This is the instantiated model of the :class:`.ViveknSentimentApproach`. For
    training your own model, please see the documentation of that class.

    The analyzer requires sentence boundaries to give a score in context.
    Tokenization is needed to make sure tokens are within bounds. Transitivity
    requirements are also required.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``SENTIMENT``
    ====================== ======================

    Parameters
    ----------
    None

    References
    ----------
    The algorithm is based on the paper `"Fast and accurate sentiment
    classification using an enhanced Naive Bayes model"
    <https://arxiv.org/abs/1305.6143>`__.

    https://github.com/vivekn/sentiment/
    """
    name = "ViveknSentimentModel"

    inputAnnotatorTypes = [AnnotatorType.TOKEN, AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTIMENT

    importantFeatureRatio = Param(Params._dummy(),
                                  "importantFeatureRatio",
                                  "proportion of feature content to be considered relevant. Defaults to 0.5",
                                  typeConverter=TypeConverters.toFloat)

    unimportantFeatureStep = Param(Params._dummy(),
                                   "unimportantFeatureStep",
                                   "proportion to lookahead in unimportant features. Defaults to 0.025",
                                   typeConverter=TypeConverters.toFloat)

    featureLimit = Param(Params._dummy(),
                         "featureLimit",
                         "content feature limit, to boost performance in very dirt text. Default disabled with -1",
                         typeConverter=TypeConverters.toInt)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel", java_model=None):
        super(ViveknSentimentModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="sentiment_vivekn", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sentiment_vivekn"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        ViveknSentimentModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ViveknSentimentModel, name, lang, remote_loc)
