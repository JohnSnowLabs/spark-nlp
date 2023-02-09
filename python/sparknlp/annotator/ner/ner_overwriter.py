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
"""Contains classes for the NerOverwriter."""

from sparknlp.common import *


class NerOverwriter(AnnotatorModel):
    """Overwrites entities of specified strings.

    The input for this Annotator have to be entities that are already extracted,
    Annotator type ``NAMED_ENTITY``. The strings specified with
    :meth:`.NerOverwriter.setStopWords` will have new entities assigned to, specified
    with :meth:`.NerOverwriter.setNewResult`.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``NAMED_ENTITY``       ``NAMED_ENTITY``
    ====================== ======================

    Parameters
    ----------
    nerWords
        The words to be overwritten
    newNerEntity
        new NER class to apply to those stopwords, by default I-OVERWRITE

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    First extract the prerequisite Entities

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
    ...     .setOutputCol("bert")
    >>> nerTagger = NerDLModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "bert"]) \\
    ...     .setOutputCol("ner")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     embeddings,
    ...     nerTagger
    ... ])
    >>> data = spark.createDataFrame([["Spark NLP Crosses Five Million Downloads, John Snow Labs Announces."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(ner)").show(truncate=False)
    +------------------------------------------------------+
    |col                                                   |
    +------------------------------------------------------+
    |[named_entity, 0, 4, B-ORG, [word -> Spark], []]      |
    |[named_entity, 6, 8, I-ORG, [word -> NLP], []]        |
    |[named_entity, 10, 16, O, [word -> Crosses], []]      |
    |[named_entity, 18, 21, O, [word -> Five], []]         |
    |[named_entity, 23, 29, O, [word -> Million], []]      |
    |[named_entity, 31, 39, O, [word -> Downloads], []]    |
    |[named_entity, 40, 40, O, [word -> ,], []]            |
    |[named_entity, 42, 45, B-ORG, [word -> John], []]     |
    |[named_entity, 47, 50, I-ORG, [word -> Snow], []]     |
    |[named_entity, 52, 55, I-ORG, [word -> Labs], []]     |
    |[named_entity, 57, 65, I-ORG, [word -> Announces], []]|
    |[named_entity, 66, 66, O, [word -> .], []]            |
    +------------------------------------------------------+

    The recognized entities can then be overwritten

    >>> nerOverwriter = NerOverwriter() \\
    ...     .setInputCols(["ner"]) \\
    ...     .setOutputCol("ner_overwritten") \\
    ...     .setNerWords(["Million"]) \\
    ...     .setNewNerEntity("B-CARDINAL")
    >>> nerOverwriter.transform(result).selectExpr("explode(ner_overwritten)").show(truncate=False)
    +---------------------------------------------------------+
    |col                                                      |
    +---------------------------------------------------------+
    |[named_entity, 0, 4, B-ORG, [word -> Spark], []]         |
    |[named_entity, 6, 8, I-ORG, [word -> NLP], []]           |
    |[named_entity, 10, 16, O, [word -> Crosses], []]         |
    |[named_entity, 18, 21, O, [word -> Five], []]            |
    |[named_entity, 23, 29, B-CARDINAL, [word -> Million], []]|
    |[named_entity, 31, 39, O, [word -> Downloads], []]       |
    |[named_entity, 40, 40, O, [word -> ,], []]               |
    |[named_entity, 42, 45, B-ORG, [word -> John], []]        |
    |[named_entity, 47, 50, I-ORG, [word -> Snow], []]        |
    |[named_entity, 52, 55, I-ORG, [word -> Labs], []]        |
    |[named_entity, 57, 65, I-ORG, [word -> Announces], []]   |
    |[named_entity, 66, 66, O, [word -> .], []]               |
    +---------------------------------------------------------+
    """
    name = "NerOverwriter"

    inputAnnotatorTypes = [AnnotatorType.NAMED_ENTITY]

    outputAnnotatorType = AnnotatorType.NAMED_ENTITY

    @keyword_only
    def __init__(self):
        super(NerOverwriter, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.NerOverwriter")
        self._setDefault(
            newNerEntity="I-OVERWRITE"
        )

    nerWords = Param(Params._dummy(), "nerWords", "The words to be overwritten",
                      typeConverter=TypeConverters.toListString)
    newNerEntity = Param(Params._dummy(), "newNerEntity", "new NER class to apply to those stopwords",
                      typeConverter=TypeConverters.toString)
    replaceEntities = Param(Params._dummy(), "replaceEntities", "Ner tags to be replaced",
                            typeConverter=TypeConverters.identity)

    def setNerWords(self, value):
        """Sets the words to be overwritten.

        Parameters
        ----------
        value : List[str]
            The words to be overwritten
        """
        return self._set(nerWords=value)

    def setNewNerEntity(self, value):
        """Sets new NER class to apply to those stopwords, by default
        I-OVERWRITE.

        Parameters
        ----------
        value : str
            NER class to apply the stopwords to
        """
        return self._set(newNerEntity=value)

    def setReplaceEntities(self, rw):
        """Sets weights dictionary with the tags that you want to replace.

        Parameters
        ----------
        rw : Dict[str, str]
        Sets weights dictionary with the tags that you want to replace...
        """
        self._call_java('setReplaceEntities', rw)
        return self
