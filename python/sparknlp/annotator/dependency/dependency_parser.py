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
"""Contains classes for the DependencyParser."""

from sparknlp.common import *


class DependencyParserApproach(AnnotatorApproach):
    """Trains an unlabeled parser that finds a grammatical relations between two
    words in a sentence.

    For instantiated/pretrained models, see :class:`.DependencyParserModel`.

    Dependency parser provides information about word relationship. For example,
    dependency parsing can tell you what the subjects and objects of a verb are,
    as well as which words are modifying (describing) the subject. This can help
    you find precise answers to specific questions.

    The required training data can be set in two different ways (only one can be
    chosen for a particular model):

    - Dependency treebank in the
      `Penn Treebank format <http://www.nltk.org/nltk_data/>`__ set with
      ``setDependencyTreeBank``
    - Dataset in the
      `CoNLL-U format <https://universaldependencies.org/format.html>`__ set
      with ``setConllU``

    Apart from that, no additional training data is needed.

    ======================== ======================
    Input Annotation types   Output Annotation type
    ======================== ======================
    ``DOCUMENT, POS, TOKEN`` ``DEPENDENCY``
    ======================== ======================

    Parameters
    ----------
    dependencyTreeBank
        Dependency treebank source files
    conllU
        Universal Dependencies source files
    numberOfIterations
        Number of iterations in training, converges to better accuracy,
        by default 10

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
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")
    >>> dependencyParserApproach = DependencyParserApproach() \\
    ...     .setInputCols(["sentence", "pos", "token"]) \\
    ...     .setOutputCol("dependency") \\
    ...     .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     posTagger,
    ...     dependencyParserApproach
    ... ])
    >>> emptyDataSet = spark.createDataFrame([[""]]).toDF("text")
    >>> pipelineModel = pipeline.fit(emptyDataSet)

    Additional training data is not needed, the dependency parser relies on the
    dependency tree bank / CoNLL-U only.

    See Also
    --------
    TypedDependencyParserApproach : to extract labels for the dependencies
    """

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.POS, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.DEPENDENCY

    dependencyTreeBank = Param(Params._dummy(),
                               "dependencyTreeBank",
                               "Dependency treebank source files",
                               typeConverter=TypeConverters.identity)

    conllU = Param(Params._dummy(),
                   "conllU",
                   "Universal Dependencies source files",
                   typeConverter=TypeConverters.identity)

    numberOfIterations = Param(Params._dummy(),
                               "numberOfIterations",
                               "Number of iterations in training, converges to better accuracy",
                               typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(DependencyParserApproach,
              self).__init__(classname="com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach")
        self._setDefault(numberOfIterations=10)

    def setNumberOfIterations(self, value):
        """Sets number of iterations in training, converges to better accuracy,
        by default 10.

        Parameters
        ----------
        value : int
            Number of iterations
        """
        return self._set(numberOfIterations=value)

    def setDependencyTreeBank(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        """Sets dependency treebank source files.

        Parameters
        ----------
        path : str
            Path to the source files
        read_as : str, optional
            How to read the file, by default ReadAs.TEXT
        options : dict, optional
            Options to read the resource, by default {"key": "value"}
        """
        opts = options.copy()
        return self._set(dependencyTreeBank=ExternalResource(path, read_as, opts))

    def setConllU(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        """Sets Universal Dependencies source files.

        Parameters
        ----------
        path : str
            Path to the source files
        read_as : str, optional
            How to read the file, by default ReadAs.TEXT
        options : dict, optional
            Options to read the resource, by default {"key": "value"}
        """
        opts = options.copy()
        return self._set(conllU=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return DependencyParserModel(java_model=java_model)


class DependencyParserModel(AnnotatorModel):
    """Unlabeled parser that finds a grammatical relation between two words in a
    sentence.

    Dependency parser provides information about word relationship. For example,
    dependency parsing can tell you what the subjects and objects of a verb are,
    as well as which words are modifying (describing) the subject. This can help
    you find precise answers to specific questions.

    This is the instantiated model of the :class:`.DependencyParserApproach`.
    For training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> dependencyParserApproach = DependencyParserModel.pretrained() \\
    ...     .setInputCols(["sentence", "pos", "token"]) \\
    ...     .setOutputCol("dependency")


    The default model is ``"dependency_conllu"``, if no name is provided.
    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models>`__.

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/graph-extraction/graph_extraction_intro.ipynb>`__.

    ================================ ======================
    Input Annotation types           Output Annotation type
    ================================ ======================
    ``[String]DOCUMENT, POS, TOKEN`` ``DEPENDENCY``
    ================================ ======================

    Parameters
    ----------
    perceptron
        Dependency parsing perceptron features

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
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")
    >>> dependencyParser = DependencyParserModel.pretrained() \\
    ...     .setInputCols(["sentence", "pos", "token"]) \\
    ...     .setOutputCol("dependency")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     posTagger,
    ...     dependencyParser
    ... ])
    >>> data = spark.createDataFrame([[
    ...     "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
    ...     "firm Federal Mogul."
    ... ]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(arrays_zip(token.result, dependency.result)) as cols") \\
    ...     .selectExpr("cols['0'] as token", "cols['1'] as dependency").show(8, truncate = False)
    +------------+------------+
    |token       |dependency  |
    +------------+------------+
    |Unions      |ROOT        |
    |representing|workers     |
    |workers     |Unions      |
    |at          |Turner      |
    |Turner      |workers     |
    |Newall      |say         |
    |say         |Unions      |
    |they        |disappointed|
    +------------+------------+

    See Also
    --------
    TypedDependencyParserMdoel : to extract labels for the dependencies
    """
    name = "DependencyParserModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.POS, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.DEPENDENCY

    perceptron = Param(Params._dummy(),
                       "perceptron",
                       "Dependency parsing perceptron features",
                       typeConverter=TypeConverters.identity)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel", java_model=None):
        super(DependencyParserModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="dependency_conllu", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "dependency_conllu"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        DependencyParserModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(DependencyParserModel, name, lang, remote_loc)
