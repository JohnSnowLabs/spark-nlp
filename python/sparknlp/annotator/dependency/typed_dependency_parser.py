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
"""Contains classes for the TypedDependencyParser."""


from sparknlp.common import *


class TypedDependencyParserApproach(AnnotatorApproach):
    """Labeled parser that finds a grammatical relation between two words in a
    sentence. Its input is either a CoNLL2009 or ConllU dataset.

    For instantiated/pretrained models, see
    :class:`.TypedDependencyParserModel`.

    Dependency parsers provide information about word relationship. For example,
    dependency parsing can tell you what the subjects and objects of a verb are,
    as well as which words are modifying (describing) the subject. This can help
    you find precise answers to specific questions.

    The parser requires the dependant tokens beforehand with e.g.
    DependencyParser. The required training data can be set in two different
    ways (only one can be chosen for a particular model):

    - Dataset in the `CoNLL 2009 format
      <https://ufal.mff.cuni.cz/conll2009-st/trial-data.html>`__ set with
      :meth:`.setConll2009`
    - Dataset in the `CoNLL-U format
      <https://universaldependencies.org/format.html>`__ set with
      :meth:`.setConllU`

    Apart from that, no additional training data is needed.

    ========================== ======================
    Input Annotation types     Output Annotation type
    ========================== ======================
    ``TOKEN, POS, DEPENDENCY`` ``LABELED_DEPENDENCY``
    ========================== ======================

    Parameters
    ----------
    conll2009
        Path to file with CoNLL 2009 format
    conllU
        Universal Dependencies source files
    numberOfIterations
        Number of iterations in training, converges to better accuracy

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
    >>> typedDependencyParser = TypedDependencyParserApproach() \\
    ...     .setInputCols(["dependency", "pos", "token"]) \\
    ...     .setOutputCol("dependency_type") \\
    ...     .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt") \\
    ...     .setNumberOfIterations(1)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     posTagger,
    ...     dependencyParser,
    ...     typedDependencyParser
    ... ])

    Additional training data is not needed, the dependency parser relies on
    CoNLL-U only.

    >>> emptyDataSet = spark.createDataFrame([[""]]).toDF("text")
    >>> pipelineModel = pipeline.fit(emptyDataSet)
    """

    inputAnnotatorTypes = [AnnotatorType.TOKEN, AnnotatorType.POS, AnnotatorType.DEPENDENCY]

    outputAnnotatorType = AnnotatorType.LABELED_DEPENDENCY

    conll2009 = Param(Params._dummy(),
                      "conll2009",
                      "Path to file with CoNLL 2009 format",
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
        super(TypedDependencyParserApproach,
              self).__init__(classname="com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach")

    def setConll2009(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        """Sets path to file with CoNLL 2009 format.

        Parameters
        ----------
        path : str
            Path to the resource
        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {"key": "value"}
        """
        opts = options.copy()
        return self._set(conll2009=ExternalResource(path, read_as, opts))

    def setConllU(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        """Sets path to Universal Dependencies source files.

        Parameters
        ----------
        path : str
            Path to the resource
        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {"key": "value"}
        """
        opts = options.copy()
        return self._set(conllU=ExternalResource(path, read_as, opts))

    def setNumberOfIterations(self, value):
        """Sets Number of iterations in training, converges to better accuracy.

        Parameters
        ----------
        value : int
            Number of iterations in training

        Returns
        -------
        [type]
            [description]
        """
        return self._set(numberOfIterations=value)

    def _create_model(self, java_model):
        return TypedDependencyParserModel(java_model=java_model)


class TypedDependencyParserModel(AnnotatorModel):
    """Labeled parser that finds a grammatical relation between two words in a
    sentence. Its input is either a CoNLL2009 or ConllU dataset.

    Dependency parsers provide information about word relationship. For example,
    dependency parsing can tell you what the subjects and objects of a verb are,
    as well as which words are modifying (describing) the subject. This can help
    you find precise answers to specific questions.

    The parser requires the dependant tokens beforehand with e.g.
    DependencyParser.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> typedDependencyParser = TypedDependencyParserModel.pretrained() \\
    ...     .setInputCols(["dependency", "pos", "token"]) \\
    ...     .setOutputCol("dependency_type")

    The default model is ``"dependency_typed_conllu"``, if no name is provided.
    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/graph-extraction/graph_extraction_intro.ipynb>`__.

    ========================== ======================
    Input Annotation types     Output Annotation type
    ========================== ======================
    ``TOKEN, POS, DEPENDENCY`` ``LABELED_DEPENDENCY``
    ========================== ======================

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
    >>> typedDependencyParser = TypedDependencyParserModel.pretrained() \\
    ...     .setInputCols(["dependency", "pos", "token"]) \\
    ...     .setOutputCol("dependency_type")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     posTagger,
    ...     dependencyParser,
    ...     typedDependencyParser
    ... ])
    >>> data = spark.createDataFrame([[
    ...     "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
    ...       "firm Federal Mogul."
    ... ]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(arrays_zip(token.result, dependency.result, dependency_type.result)) as cols") \\
    ...     .selectExpr("cols['0'] as token", "cols['1'] as dependency", "cols['2'] as dependency_type") \\
    ...     .show(8, truncate = False)
    +------------+------------+---------------+
    |token       |dependency  |dependency_type|
    +------------+------------+---------------+
    |Unions      |ROOT        |root           |
    |representing|workers     |amod           |
    |workers     |Unions      |flat           |
    |at          |Turner      |case           |
    |Turner      |workers     |flat           |
    |Newall      |say         |nsubj          |
    |say         |Unions      |parataxis      |
    |they        |disappointed|nsubj          |
    +------------+------------+---------------+
    """

    name = "TypedDependencyParserModel"

    inputAnnotatorTypes = [AnnotatorType.TOKEN, AnnotatorType.POS, AnnotatorType.DEPENDENCY]

    outputAnnotatorType = AnnotatorType.LABELED_DEPENDENCY

    trainOptions = Param(Params._dummy(),
                         "trainOptions",
                         "Training Options",
                         typeConverter=TypeConverters.identity)

    trainParameters = Param(Params._dummy(),
                            "trainParameters",
                            "Training Parameters",
                            typeConverter=TypeConverters.identity)

    trainDependencyPipe = Param(Params._dummy(),
                                "trainDependencyPipe",
                                "Training dependency pipe",
                                typeConverter=TypeConverters.identity)

    conllFormat = Param(Params._dummy(),
                        "conllFormat",
                        "CoNLL Format",
                        typeConverter=TypeConverters.toString)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel",
                 java_model=None):
        super(TypedDependencyParserModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="dependency_typed_conllu", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "dependency_typed_conllu"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        TypedDependencyParserModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TypedDependencyParserModel, name, lang, remote_loc)
