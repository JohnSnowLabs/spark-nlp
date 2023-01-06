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
"""Contains classes for GraphExtraction."""
from sparknlp.common import *


class GraphExtraction(AnnotatorModel):
    """Extracts a dependency graph between entities.

    The GraphExtraction class takes e.g. extracted entities from a
    :class:`.NerDLModel` and creates a dependency tree which describes how the
    entities relate to each other. For that a triple store format is used. Nodes
    represent the entities and the edges represent the relations between those
    entities. The graph can then be used to find relevant relationships between
    words.

    Both the :class:`.DependencyParserModel` and
    :class:`.TypedDependencyParserModel` need to be
    present in the pipeline. There are two ways to set them:

    #. Both Annotators are present in the pipeline already. The dependencies are
       taken implicitly from these two Annotators.
    #. Setting :meth:`.setMergeEntities` to ``True`` will download the
       default pretrained models for those two Annotators automatically. The
       specific models can also be set with :meth:`.setDependencyParserModel`
       and :meth:`.setTypedDependencyParserModel`:

        >>> graph_extraction = GraphExtraction() \\
        ...     .setInputCols(["document", "token", "ner"]) \\
        ...     .setOutputCol("graph") \\
        ...     .setRelationshipTypes(["prefer-LOC"]) \\
        ...     .setMergeEntities(True)
        >>>     #.setDependencyParserModel(["dependency_conllu", "en",  "public/models"])
        >>>     #.setTypedDependencyParserModel(["dependency_typed_conllu", "en",  "public/models"])

    ================================= ======================
    Input Annotation types            Output Annotation type
    ================================= ======================
    ``DOCUMENT, TOKEN, NAMED_ENTITY`` ``NODE``
    ================================= ======================

    Parameters
    ----------
    relationshipTypes
        Paths to find between a pair of token and entity
    entityTypes
        Paths to find between a pair of entities
    explodeEntities
        When set to true find paths between entities
    rootTokens
        Tokens to be consider as root to start traversing the paths. Use it
        along with explodeEntities
    maxSentenceSize
        Maximum sentence size that the annotator will process, by default 1000.
        Above this, the sentence is skipped
    minSentenceSize
        Minimum sentence size that the annotator will process, by default 2.
        Below this, the sentence is skipped.
    mergeEntities
        Merge same neighboring entities as a single token
    includeEdges
        Whether to include edges when building paths
    delimiter
        Delimiter symbol used for path output
    posModel
        Coordinates (name, lang, remoteLoc) to a pretrained POS model
    dependencyParserModel
        Coordinates (name, lang, remoteLoc) to a pretrained Dependency Parser
        model
    typedDependencyParserModel
        Coordinates (name, lang, remoteLoc) to a pretrained Typed Dependency
        Parser model

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
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("embeddings")
    >>> nerTagger = NerDLModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "embeddings"]) \\
    ...     .setOutputCol("ner")
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")
    >>> dependencyParser = DependencyParserModel.pretrained() \\
    ...     .setInputCols(["sentence", "pos", "token"]) \\
    ...     .setOutputCol("dependency")
    >>> typedDependencyParser = TypedDependencyParserModel.pretrained() \\
    ...     .setInputCols(["dependency", "pos", "token"]) \\
    ...     .setOutputCol("dependency_type")
    >>> graph_extraction = GraphExtraction() \\
    ...     .setInputCols(["document", "token", "ner"]) \\
    ...     .setOutputCol("graph") \\
    ...     .setRelationshipTypes(["prefer-LOC"])
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     embeddings,
    ...     nerTagger,
    ...     posTagger,
    ...     dependencyParser,
    ...     typedDependencyParser,
    ...     graph_extraction
    ... ])
    >>> data = spark.createDataFrame([["You and John prefer the morning flight through Denver"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("graph").show(truncate=False)
    +-----------------------------------------------------------------------------------------------------------------+
    |graph                                                                                                            |
    +-----------------------------------------------------------------------------------------------------------------+
    |[[node, 13, 18, prefer, [relationship -> prefer,LOC, path1 -> prefer,nsubj,morning,flat,flight,flat,Denver], []]]|
    +-----------------------------------------------------------------------------------------------------------------+

    See Also
    --------
    GraphFinisher : to output the paths in a more generic format, like RDF
    """
    name = "GraphExtraction"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN, AnnotatorType.NAMED_ENTITY]

    optionalInputAnnotatorTypes = [AnnotatorType.DEPENDENCY, AnnotatorType.LABELED_DEPENDENCY]

    outputAnnotatorType = AnnotatorType.NODE

    relationshipTypes = Param(Params._dummy(),
                              "relationshipTypes",
                              "Find paths between a pair of token and entity",
                              typeConverter=TypeConverters.toListString)

    entityTypes = Param(Params._dummy(),
                        "entityTypes",
                        "Find paths between a pair of entities",
                        typeConverter=TypeConverters.toListString)

    explodeEntities = Param(Params._dummy(),
                            "explodeEntities",
                            "When set to true find paths between entities",
                            typeConverter=TypeConverters.toBoolean)

    rootTokens = Param(Params._dummy(),
                       "rootTokens",
                       "Tokens to be consider as root to start traversing the paths. Use it along with explodeEntities",
                       typeConverter=TypeConverters.toListString)

    maxSentenceSize = Param(Params._dummy(),
                            "maxSentenceSize",
                            "Maximum sentence size that the annotator will process. Above this, the sentence is skipped",
                            typeConverter=TypeConverters.toInt)

    minSentenceSize = Param(Params._dummy(),
                            "minSentenceSize",
                            "Minimum sentence size that the annotator will process. Above this, the sentence is skipped",
                            typeConverter=TypeConverters.toInt)

    mergeEntities = Param(Params._dummy(),
                          "mergeEntities",
                          "Merge same neighboring entities as a single token",
                          typeConverter=TypeConverters.toBoolean)

    mergeEntitiesIOBFormat = Param(Params._dummy(),
                                   "mergeEntitiesIOBFormat",
                                   "IOB format to apply when merging entities",
                                   typeConverter=TypeConverters.toString)

    includeEdges = Param(Params._dummy(),
                         "includeEdges",
                         "Whether to include edges when building paths",
                         typeConverter=TypeConverters.toBoolean)

    delimiter = Param(Params._dummy(),
                      "delimiter",
                      "Delimiter symbol used for path output",
                      typeConverter=TypeConverters.toString)

    posModel = Param(Params._dummy(),
                     "posModel",
                     "Coordinates (name, lang, remoteLoc) to a pretrained POS model",
                     typeConverter=TypeConverters.toListString)

    dependencyParserModel = Param(Params._dummy(),
                                  "dependencyParserModel",
                                  "Coordinates (name, lang, remoteLoc) to a pretrained Dependency Parser model",
                                  typeConverter=TypeConverters.toListString)

    typedDependencyParserModel = Param(Params._dummy(),
                                       "typedDependencyParserModel",
                                       "Coordinates (name, lang, remoteLoc) to a pretrained Typed Dependency Parser model",
                                       typeConverter=TypeConverters.toListString)

    def setRelationshipTypes(self, value):
        """Sets paths to find between a pair of token and entity.

        Parameters
        ----------
        value : List[str]
            Paths to find between a pair of token and entity
        """
        return self._set(relationshipTypes=value)

    def setEntityTypes(self, value):
        """Sets paths to find between a pair of entities.

        Parameters
        ----------
        value : List[str]
            Paths to find between a pair of entities
        """
        return self._set(entityTypes=value)

    def setExplodeEntities(self, value):
        """Sets whether to find paths between entities.

        Parameters
        ----------
        value : bool
            Whether to find paths between entities.
        """
        return self._set(explodeEntities=value)

    def setRootTokens(self, value):
        """Sets tokens to be considered as the root to start traversing the paths.

        Use it along with explodeEntities.

        Parameters
        ----------
        value : List[str]
            Sets Tokens to be consider as root to start traversing the paths.
        """
        return self._set(rootTokens=value)

    def setMaxSentenceSize(self, value):
        """Sets Maximum sentence size that the annotator will process, by
        default 1000.

        Above this, the sentence is skipped.

        Parameters
        ----------
        value : int
            Maximum sentence size that the annotator will process
        """
        return self._set(maxSentenceSize=value)

    def setMinSentenceSize(self, value):
        """Sets Minimum sentence size that the annotator will process, by
        default 2.

        Below this, the sentence is skipped.

        Parameters
        ----------
        value : int
            Minimum sentence size that the annotator will process
        """
        return self._set(minSentenceSize=value)

    def setMergeEntities(self, value):
        """Sets whether to merge same neighboring entities as a single token.

        Parameters
        ----------
        value : bool
            Whether to merge same neighboring entities as a single token.
        """
        return self._set(mergeEntities=value)

    def setMergeEntitiesIOBFormat(self, value):
        """Sets IOB format to apply when merging entities.

        Parameters
        ----------
        value : str
            IOB format to apply when merging entities. Values IOB or IOB2
        """
        return self._set(mergeEntitiesIOBFormat=value)

    def setIncludeEdges(self, value):
        """Sets whether to include edges when building paths.

        Parameters
        ----------
        value : bool
            Whether to include edges when building paths
        """
        return self._set(includeEdges=value)

    def setDelimiter(self, value):
        """Sets delimiter symbol used for path output.

        Parameters
        ----------
        value : str
            Delimiter symbol used for path output
        """
        return self._set(delimiter=value)

    def setPosModel(self, value):
        """Sets Coordinates (name, lang, remoteLoc) to a pretrained POS model.

        Parameters
        ----------
        value : List[str]
            Coordinates (name, lang, remoteLoc) to a pretrained POS model
        """
        return self._set(posModel=value)

    def setDependencyParserModel(self, value):
        """Sets Coordinates (name, lang, remoteLoc) to a pretrained Dependency
        Parser model.

        Parameters
        ----------
        value : List[str]
            Coordinates (name, lang, remoteLoc) to a pretrained Dependency
            Parser model
        """
        return self._set(dependencyParserModel=value)

    def setTypedDependencyParserModel(self, value):
        """Sets Coordinates (name, lang, remoteLoc) to a pretrained Typed
        Dependency Parser model.

        Parameters
        ----------
        value : List[str]
            Coordinates (name, lang, remoteLoc) to a pretrained Typed Dependency
            Parser model
        """
        return self._set(typedDependencyParserModel=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.GraphExtraction", java_model=None):
        super(GraphExtraction, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            maxSentenceSize=1000,
            minSentenceSize=2
        )

