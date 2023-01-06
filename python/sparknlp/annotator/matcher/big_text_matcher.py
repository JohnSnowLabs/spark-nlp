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
"""Contains classes for the BigTextMatcher."""

from sparknlp.common import *
from sparknlp.annotator.matcher.text_matcher import TextMatcherModel


class BigTextMatcher(AnnotatorApproach, HasStorage):
    """Annotator to match exact phrases (by token) provided in a file against a
    Document.

    A text file of predefined phrases must be provided with ``setStoragePath``.

    In contrast to the normal ``TextMatcher``, the ``BigTextMatcher`` is
    designed for large corpora.

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
        whether to ignore case in index lookups, by default True
    mergeOverlapping
        whether to merge overlapping matched chunks, by default False
    tokenizer
        TokenizerModel to use to tokenize input file for building a Trie

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
    ...     .setInputCols("document") \\
    ...     .setOutputCol("token")
    >>> data = spark.createDataFrame([["Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum"]]).toDF("text")
    >>> entityExtractor = BigTextMatcher() \\
    ...     .setInputCols("document", "token") \\
    ...     .setStoragePath("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT) \\
    ...     .setOutputCol("entity") \\
    ...     .setCaseSensitive(False)
    >>> pipeline = Pipeline().setStages([documentAssembler, tokenizer, entityExtractor])
    >>> results = pipeline.fit(data).transform(data)
    >>> results.selectExpr("explode(entity)").show(truncate=False)
    +--------------------------------------------------------------------+
    |col                                                                 |
    +--------------------------------------------------------------------+
    |[chunk, 6, 24, dolore magna aliqua, [sentence -> 0, chunk -> 0], []]|
    |[chunk, 53, 59, laborum, [sentence -> 0, chunk -> 1], []]           |
    +--------------------------------------------------------------------+
    """

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    entities = Param(Params._dummy(),
                     "entities",
                     "ExternalResource for entities",
                     typeConverter=TypeConverters.identity)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case in index lookups",
                          typeConverter=TypeConverters.toBoolean)

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    tokenizer = Param(Params._dummy(),
                      "tokenizer",
                      "TokenizerModel to use to tokenize input file for building a Trie",
                      typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(BigTextMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.btm.BigTextMatcher")
        self._setDefault(caseSensitive=True)
        self._setDefault(mergeOverlapping=False)

    def _create_model(self, java_model):
        return TextMatcherModel(java_model=java_model)

    def setEntities(self, path, read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets ExternalResource for entities.

        Parameters
        ----------
        path : str
            Path to the resource
        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {"format": "text"}
        """
        return self._set(entities=ExternalResource(path, read_as, options.copy()))

    def setCaseSensitive(self, b):
        """Sets whether to ignore case in index lookups, by default True.

        Parameters
        ----------
        b : bool
            Whether to ignore case in index lookups
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

    def setTokenizer(self, tokenizer_model):
        """Sets TokenizerModel to use to tokenize input file for building a
        Trie.

        Parameters
        ----------
        tokenizer_model : :class:`TokenizerModel <sparknlp.annotator.TokenizerModel>`
            TokenizerModel to use to tokenize input file

        """
        tokenizer_model._transfer_params_to_java()
        return self._set(tokenizer_model._java_obj)


class BigTextMatcherModel(AnnotatorModel, HasStorageModel):
    """Instantiated model of the BigTextMatcher.

    This is the instantiated model of the :class:`.BigTextMatcher`.
    For training your own model, please see the documentation of that class.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    caseSensitive
        Whether to ignore case in index lookups
    mergeOverlapping
        Whether to merge overlapping matched chunks, by default False
    searchTrie
        SearchTrie
    """
    name = "BigTextMatcherModel"
    databases = ['TMVOCAB', 'TMEDGES', 'TMNODES']

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case in index lookups",
                          typeConverter=TypeConverters.toBoolean)

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    searchTrie = Param(Params._dummy(),
                       "searchTrie",
                       "searchTrie",
                       typeConverter=TypeConverters.identity)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.btm.TextMatcherModel", java_model=None):
        super(BigTextMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    def setMergeOverlapping(self, b):
        """Sets whether to merge overlapping matched chunks, by default False.

        Parameters
        ----------
        v : bool
            Whether to merge overlapping matched chunks, by default False
        """
        return self._set(mergeOverlapping=b)

    def setCaseSensitive(self, v):
        """Sets whether to ignore case in index lookups.

        Parameters
        ----------
        b : bool
            Whether to ignore case in index lookups
        """
        return self._set(caseSensitive=v)

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

    @staticmethod
    def loadStorage(path, spark, storage_ref):
        """Loads the model from storage.

        Parameters
        ----------
        path : str
            Path to the model
        spark : :class:`pyspark.sql.SparkSession`
            The current SparkSession
        storage_ref : str
            Identifiers for the model parameters
        """
        HasStorageModel.loadStorages(path, spark, storage_ref, BigTextMatcherModel.databases)

