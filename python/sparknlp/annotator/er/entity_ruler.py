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
"""Contains classes for the EntityRuler."""

from sparknlp.common import *


class EntityRulerApproach(AnnotatorApproach, HasStorage):
    """Fits an Annotator to match exact strings or regex patterns provided in a
    file against a Document and assigns them an named entity. The definitions
    can contain any number of named entities.

    There are multiple ways and formats to set the extraction resource. It is
    possible to set it either as a "JSON", "JSONL" or "CSV" file. A path to the
    file needs to be provided to ``setPatternsResource``. The file format needs
    to be set as the "format" field in the ``option`` parameter map and
    depending on the file type, additional parameters might need to be set.

    If the file is in a JSON format, then the rule definitions need to be given
    in a list with the fields "id", "label" and "patterns"::

         [
            {
              "id": "person-regex",
              "label": "PERSON",
              "patterns": ["\\w+\\s\\w+", "\\w+-\\w+"]
            },
            {
              "id": "locations-words",
              "label": "LOCATION",
              "patterns": ["Winterfell"]
            }
        ]

    The same fields also apply to a file in the JSONL format::

        {"id": "names-with-j", "label": "PERSON", "patterns": ["Jon", "John", "John Snow"]}
        {"id": "names-with-s", "label": "PERSON", "patterns": ["Stark", "Snow"]}
        {"id": "names-with-e", "label": "PERSON", "patterns": ["Eddard", "Eddard Stark"]}

    In order to use a CSV file, an additional parameter "delimiter" needs to be
    set. In this case, the delimiter might be set by using
    ``.setPatternsResource("patterns.csv", ReadAs.TEXT, {"format": "csv", "delimiter": "|")})``::

        PERSON|Jon
        PERSON|John
        PERSON|John Snow
        LOCATION|Winterfell

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    patternsResource
        Resource in JSON or CSV format to map entities to patterns
    useStorage
        Whether to use RocksDB storage to serialize patterns

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from sparknlp.common import *
    >>> from pyspark.ml import Pipeline

    In this example, the entities file as the form of::

        PERSON|Jon
        PERSON|John
        PERSON|John Snow
        LOCATION|Winterfell

    where each line represents an entity and the associated string delimited by "|".

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> entityRuler = EntityRulerApproach() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("entities") \\
    ...     .setPatternsResource(
    ...       "patterns.csv",
    ...       ReadAs.TEXT,
    ...       {"format": "csv", "delimiter": "\\\\|"}
    ...     )
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     entityRuler
    ... ])
    >>> data = spark.createDataFrame([["Jon Snow wants to be lord of Winterfell."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(entities)").show(truncate=False)
    +--------------------------------------------------------------------+
    |col                                                                 |
    +--------------------------------------------------------------------+
    |[chunk, 0, 2, Jon, [entity -> PERSON, sentence -> 0], []]           |
    |[chunk, 29, 38, Winterfell, [entity -> LOCATION, sentence -> 0], []]|
    +--------------------------------------------------------------------+
    """
    name = "EntityRulerApproach"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]
    optionalInputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    patternsResource = Param(Params._dummy(),
                             "patternsResource",
                             "Resource in JSON or CSV format to map entities to patterns",
                             typeConverter=TypeConverters.identity)

    useStorage = Param(Params._dummy(),
                       "useStorage",
                       "Whether to use RocksDB storage to serialize patterns",
                       typeConverter=TypeConverters.toBoolean)

    sentenceMatch = Param(Params._dummy(),
                          "sentenceMatch",
                          "Whether to find match at sentence level. True: sentence level. False: token level",
                          typeConverter=TypeConverters.toBoolean)

    alphabet = Param(Params._dummy(),
                     "alphabet",
                     "Alphabet resource path to plain text file with all characters in a given alphabet",
                     typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(EntityRulerApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.er.EntityRulerApproach")

    def setPatternsResource(self, path, read_as=ReadAs.TEXT, options={"format": "JSON"}):
        """Sets Resource in JSON or CSV format to map entities to patterns.

        Parameters
        ----------
        path : str
            Path to the resource
        read_as : str, optional
            How to interpret the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for parsing the resource, by default {"format": "JSON"}
        """
        return self._set(patternsResource=ExternalResource(path, read_as, options))

    def setUseStorage(self, value):
        """Sets whether to use RocksDB storage to serialize patterns.

        Parameters
        ----------
        value : bool
            Whether to use RocksDB storage to serialize patterns.
        """
        return self._set(useStorage=value)

    def setSentenceMatch(self, value):
        """Sets whether to find match at sentence level.

        Parameters
        ----------
        value : bool
            True: sentence level. False: token level
        """
        return self._set(sentenceMatch=value)

    def setAlphabetResource(self, path):
        """Alphabet Resource (a simple plain text with all language characters)

        Parameters
        ----------
        path : str
            Path to the resource
        """
        return self._set(alphabet=ExternalResource(path, read_as=ReadAs.TEXT, options={}))

    def _create_model(self, java_model):
        return EntityRulerModel(java_model=java_model)


class EntityRulerModel(AnnotatorModel, HasStorageModel):
    """Instantiated model of the EntityRulerApproach.
    For usage and examples see the documentation of the main class.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CHUNK``
    ====================== ======================
    """
    name = "EntityRulerModel"
    database = ['ENTITY_PATTERNS']

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]
    optionalInputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.er.EntityRulerModel", java_model=None):
        super(EntityRulerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name, lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(EntityRulerModel, name, lang, remote_loc)

    @staticmethod
    def loadStorage(path, spark, storage_ref):
        HasStorageModel.loadStorages(path, spark, storage_ref, EntityRulerModel.database)

