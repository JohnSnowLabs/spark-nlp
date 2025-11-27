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
"""Contains classes for Word2Vec."""


from sparknlp.common import *


class Word2VecApproach(AnnotatorApproach, HasStorageRef, HasEnableCachingProperties):
    """Trains a Word2Vec model that creates vector representations of words in a
    text corpus.

    The algorithm first constructs a vocabulary from the corpus and then learns
    vector representation of words in the vocabulary. The vector representation
    can be used as features in natural language processing and machine learning
    algorithms.

    We use Word2Vec implemented in Spark ML. It uses skip-gram model in our
    implementation and a hierarchical softmax method to train the model. The
    variable names in the implementation match the original C implementation.

    For instantiated/pretrained models, see :class:`.Word2VecModel`.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models>`__.

    ====================== =======================
    Input Annotation types Output Annotation type
    ====================== =======================
    ``TOKEN``              ``WORD_EMBEDDINGS``
    ====================== =======================

    Parameters
    ----------
    vectorSize
        The dimension of codes after transforming from words (> 0), by default
        100
    windowSize
        The window size (context words from [-window, window]) (> 0), by default
        5
    numPartitions
        Number of partitions for sentences of words (> 0), by default 1
    minCount
        The minimum number of times a token must appear to be included in the
        word2vec model's vocabulary (>= 0), by default 1
    maxSentenceLength
        The window size (Maximum length (in words) of each sentence in the input
        data. Any sentence longer than this threshold will be divided into
        chunks up to the size (> 0), by default 1000
    stepSize
        Step size (learning rate) to be used for each iteration of optimization
        (> 0), by default 0.025
    maxIter
        Maximum number of iterations (>= 0), by default 1
    seed
        Random seed, by default 44


    References
    ----------
    For the original C implementation, see https://code.google.com/p/word2vec/

    For the research paper, see `Efficient Estimation of Word Representations in
    Vector Space <https://arxiv.org/abs/1301.3781>`__ and `Distributed
    Representations of Words and Phrases and their Compositionality
    <https://arxiv.org/pdf/1310.4546v1.pdf>`__.

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
    >>> embeddings = Word2VecApproach() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("embeddings")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings
    ...     ])
    >>> path = "sherlockholmes.txt"
    >>> dataset = spark.read.text(path).toDF("text")
    >>> pipelineModel = pipeline.fit(dataset)
    """
    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.WORD_EMBEDDINGS

    vectorSize = Param(Params._dummy(),
                       "vectorSize",
                       "the dimension of codes after transforming from words (> 0)",
                       typeConverter=TypeConverters.toInt)

    windowSize = Param(Params._dummy(),
                       "windowSize",
                       "the window size (context words from [-window, window]) (> 0)",
                       typeConverter=TypeConverters.toInt)

    numPartitions = Param(Params._dummy(),
                          "numPartitions",
                          "number of partitions for sentences of words (> 0)",
                          typeConverter=TypeConverters.toInt)

    minCount = Param(Params._dummy(),
                     "minCount",
                     "the minimum number of times a token must " +
                     "appear to be included in the word2vec model's vocabulary (>= 0)",
                     typeConverter=TypeConverters.toInt)

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "the window size (Maximum length (in words) of each sentence in the input data. Any sentence longer than this threshold will " +
                              "be divided into chunks up to the size (> 0)",
                              typeConverter=TypeConverters.toInt)

    stepSize = Param(Params._dummy(),
                     "stepSize",
                     "Step size (learning rate) to be used for each iteration of optimization (> 0)",
                     typeConverter=TypeConverters.toFloat)

    maxIter = Param(Params._dummy(),
                    "maxIter",
                    "maximum number of iterations (>= 0)",
                    typeConverter=TypeConverters.toInt)

    seed = Param(Params._dummy(),
                 "seed",
                 "Random seed",
                 typeConverter=TypeConverters.toInt)

    def setVectorSize(self, vectorSize):
        """
        Sets vector size (default: 100).
        """
        return self._set(vectorSize=vectorSize)

    def setWindowSize(self, windowSize):
        """
        Sets window size (default: 5).
        """
        return self._set(windowSize=windowSize)

    def setStepSize(self, stepSize):
        """
        Sets initial learning rate (default: 0.025).
        """
        return self._set(stepSize=stepSize)

    def setNumPartitions(self, numPartitions):
        """
        Sets number of partitions (default: 1). Use a small number for
        accuracy.
        """
        return self._set(numPartitions=numPartitions)

    def setMaxIter(self, numIterations):
        """
        Sets number of iterations (default: 1), which should be smaller
        than or equal to number of partitions.
        """
        return self._set(maxIter=numIterations)

    def setSeed(self, seed):
        """
        Sets random seed.
        """
        return self._set(seed=seed)

    def setMinCount(self, minCount):
        """
        Sets minCount, the minimum number of times a token must appear
        to be included in the word2vec model's vocabulary (default: 5).
        """
        return self._set(minCount=minCount)

    def setMaxSentenceLength(self, maxSentenceLength):
        """
        Maximum length (in words) of each sentence in the input data.
        Any sentence longer than this threshold will be divided into
        chunks up to the size (> 0)
        """
        return self._set(maxSentenceLength=maxSentenceLength)

    @keyword_only
    def __init__(self):
        super(Word2VecApproach, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.Word2VecApproach")
        self._setDefault(
            vectorSize=100,
            windowSize=5,
            numPartitions=1,
            minCount=1,
            maxSentenceLength=1000,
            stepSize=0.025,
            maxIter=1,
            seed=44
        )

    def _create_model(self, java_model):
        return Word2VecModel(java_model=java_model)


class Word2VecModel(AnnotatorModel, HasStorageRef, HasEmbeddingsProperties):
    """Word2Vec model that creates vector representations of words in a text
    corpus.

    The algorithm first constructs a vocabulary from the corpus and then learns
    vector representation of words in the vocabulary. The vector representation
    can be used as features in natural language processing and machine learning
    algorithms.

    We use Word2Vec implemented in Spark ML. It uses skip-gram model in our
    implementation and a hierarchical softmax method to train the model. The
    variable names in the implementation match the original C implementation.

    This is the instantiated model of the :class:`.Word2VecApproach`. For
    training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = Word2VecModel.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("embeddings")

    The default model is `"word2vec_gigaword_300"`, if no name is provided.

    ====================== =======================
    Input Annotation types Output Annotation type
    ====================== =======================
    ``TOKEN``              ``WORD_EMBEDDINGS``
    ====================== =======================

    Parameters
    ----------
    vectorSize
        The dimension of codes after transforming from words (> 0), by default
        100

    References
    ----------
    For the original C implementation, see https://code.google.com/p/word2vec/

    For the research paper, see `Efficient Estimation of Word Representations in
    Vector Space <https://arxiv.org/abs/1301.3781>`__ and `Distributed
    Representations of Words and Phrases and their Compositionality
    <https://arxiv.org/pdf/1310.4546v1.pdf>`__.

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
    >>> embeddings = Word2VecModel.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[0.06222493574023247,0.011579325422644615,0.009919632226228714,0.109361454844...|
    +--------------------------------------------------------------------------------+
    """
    name = "Word2VecModel"

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.WORD_EMBEDDINGS

    vectorSize = Param(Params._dummy(),
                       "vectorSize",
                       "the dimension of codes after transforming from words (> 0)",
                       typeConverter=TypeConverters.toInt)

    def setVectorSize(self, vectorSize):
        """
        Sets vector size (default: 100).
        """
        return self._set(vectorSize=vectorSize)

    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.Word2VecModel", java_model=None):
        super(Word2VecModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            vectorSize=100
        )

    @staticmethod
    def pretrained(name="word2vec_gigaword_300", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "word2vec_wiki"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        Word2VecModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(Word2VecModel, name, lang, remote_loc)

    def getVectors(self):
        """
        Returns the vector representation of the words as a dataframe
        with two fields, word and vector.
        """
        return self._call_java("getVectors")
