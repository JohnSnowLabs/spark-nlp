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
"""Contains classes for the TextMatcher."""


from sparknlp.common import *


class TextMatcher(AnnotatorApproach):
    """Annotator to match exact phrases (by token) provided in a file against a
    Document.

    A text file of predefined phrases must be provided with
    :meth:`.setEntities`.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/text-matcher-pipeline/extractor.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    entities
        ExternalResource for entities
    caseSensitive
        Whether to match regardless of case, by default True
    mergeOverlapping
        Whether to merge overlapping matched chunks, by default False
    entityValue
        Value for the entity metadata field
    buildFromTokens
        Whether the TextMatcher should take the CHUNK from TOKEN or not

    Examples
    --------
    In this example, the entities file is of the form::

        ...
        dolore magna aliqua
        lorem ipsum dolor. sit
        laborum
        ...

    where each line represents an entity phrase to be extracted.

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
    >>> data = spark.createDataFrame([["Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum"]]).toDF("text")
    >>> entityExtractor = TextMatcher() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setEntities("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT) \\
    ...     .setOutputCol("entity") \\
    ...     .setCaseSensitive(False)
    >>> pipeline = Pipeline().setStages([documentAssembler, tokenizer, entityExtractor])
    >>> results = pipeline.fit(data).transform(data)
    >>> results.selectExpr("explode(entity) as result").show(truncate=False)
    +------------------------------------------------------------------------------------------+
    |result                                                                                    |
    +------------------------------------------------------------------------------------------+
    |[chunk, 6, 24, dolore magna aliqua, [entity -> entity, sentence -> 0, chunk -> 0], []]    |
    |[chunk, 27, 48, Lorem ipsum dolor. sit, [entity -> entity, sentence -> 0, chunk -> 1], []]|
    |[chunk, 53, 59, laborum, [entity -> entity, sentence -> 0, chunk -> 2], []]               |
    +------------------------------------------------------------------------------------------+

    See Also
    --------
    BigTextMatcher : to match large amounts of text
    """

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    entities = Param(Params._dummy(),
                     "entities",
                     "ExternalResource for entities",
                     typeConverter=TypeConverters.identity)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to match regardless of case. Defaults true",
                          typeConverter=TypeConverters.toBoolean)

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    entityValue = Param(Params._dummy(),
                        "entityValue",
                        "value for the entity metadata field",
                        typeConverter=TypeConverters.toString)

    buildFromTokens = Param(Params._dummy(),
                            "buildFromTokens",
                            "whether the TextMatcher should take the CHUNK from TOKEN or not",
                            typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(TextMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.TextMatcher")
        self._setDefault(inputCols=[AnnotatorType.DOCUMENT, AnnotatorType.TOKEN])
        self._setDefault(caseSensitive=True)
        self._setDefault(mergeOverlapping=False)

    def _create_model(self, java_model):
        return TextMatcherModel(java_model=java_model)

    def setEntities(self, path, read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets the external resource for the entities.

        Parameters
        ----------
        path : str
            Path to the external resource
        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {"format": "text"}
        """
        return self._set(entities=ExternalResource(path, read_as, options.copy()))

    def setCaseSensitive(self, b):
        """Sets whether to match regardless of case, by default True.

        Parameters
        ----------
        b : bool
            Whether to match regardless of case
        """
        return self._set(caseSensitive=b)

    def setMergeOverlapping(self, b):
        """Sets whether to merge overlapping matched chunks, by default False.

        Parameters
        ----------
        b : bool
            Whether to merge overlapping matched chunks
        """
        return self._set(mergeOverlapping=b)

    def setEntityValue(self, b):
        """Sets value for the entity metadata field.

        Parameters
        ----------
        b : str
            Value for the entity metadata field
        """
        return self._set(entityValue=b)

    def setBuildFromTokens(self, b):
        """Sets whether the TextMatcher should take the CHUNK from TOKEN or not.

        Parameters
        ----------
        b : bool
            Whether the TextMatcher should take the CHUNK from TOKEN or not
        """
        return self._set(buildFromTokens=b)


class TextMatcherModel(AnnotatorModel):
    """Instantiated model of the TextMatcher.

    This is the instantiated model of the :class:`.TextMatcher`. For training
    your own model, please see the documentation of that class.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    mergeOverlapping
        Whether to merge overlapping matched chunks, by default False
    entityValue
        Value for the entity metadata field
    buildFromTokens
        Whether the TextMatcher should take the CHUNK from TOKEN or not
    """
    name = "TextMatcherModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    searchTrie = Param(Params._dummy(),
                       "searchTrie",
                       "searchTrie",
                       typeConverter=TypeConverters.identity)

    entityValue = Param(Params._dummy(),
                        "entityValue",
                        "value for the entity metadata field",
                        typeConverter=TypeConverters.toString)

    buildFromTokens = Param(Params._dummy(),
                            "buildFromTokens",
                            "whether the TextMatcher should take the CHUNK from TOKEN or not",
                            typeConverter=TypeConverters.toBoolean)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.TextMatcherModel", java_model=None):
        super(TextMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    def setMergeOverlapping(self, b):
        """Sets whether to merge overlapping matched chunks, by default False.

        Parameters
        ----------
        b : bool
            Whether to merge overlapping matched chunks
        """
        return self._set(mergeOverlapping=b)

    def setEntityValue(self, b):
        """Sets value for the entity metadata field.

        Parameters
        ----------
        b : str
            Value for the entity metadata field
        """
        return self._set(entityValue=b)

    def setBuildFromTokens(self, b):
        """Sets whether the TextMatcher should take the CHUNK from TOKEN or not.

        Parameters
        ----------
        b : bool
            Whether the TextMatcher should take the CHUNK from TOKEN or not
        """
        return self._set(buildFromTokens=b)

    @staticmethod
    def pretrained(name, lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        TextMatcherModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TextMatcherModel, name, lang, remote_loc)
