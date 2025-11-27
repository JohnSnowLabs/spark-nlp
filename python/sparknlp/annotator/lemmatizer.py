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
"""Contains classes for the Lemmatizer."""
from sparknlp.common import *


class Lemmatizer(AnnotatorApproach):
    """Class to find lemmas out of words with the objective of returning a base
    dictionary word.

    Retrieves the significant part of a word. A dictionary of predefined lemmas
    must be provided with :meth:`.setDictionary`.

    For instantiated/pretrained models, see :class:`.LemmatizerModel`.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models?task=Lemmatization>`__.
    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/italian/Train-Lemmatizer-Italian.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    dictionary
        lemmatizer external dictionary.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    In this example, the lemma dictionary ``lemmas_small.txt`` has the form of::

        ...
        pick	->	pick	picks	picking	picked
        peck	->	peck	pecking	pecked	pecks
        pickle	->	pickle	pickles	pickled	pickling
        pepper	->	pepper	peppers	peppered	peppering
        ...

    where each key is delimited by ``->`` and values are delimited by ``\\t``

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentenceDetector = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> lemmatizer = Lemmatizer() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("lemma") \\
    ...     .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\\t")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentenceDetector,
    ...       tokenizer,
    ...       lemmatizer
    ...     ])
    >>> data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers."]]) \\
    ...     .toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("lemma.result").show(truncate=False)
    +------------------------------------------------------------------+
    |result                                                            |
    +------------------------------------------------------------------+
    |[Peter, Pipers, employees, are, pick, peck, of, pickle, pepper, .]|
    +------------------------------------------------------------------+
    """
    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "lemmatizer external dictionary." +
                       " needs 'keyDelimiter' and 'valueDelimiter' in options for parsing target text",
                       typeConverter=TypeConverters.identity)

    formCol = Param(Params._dummy(),
                    "formCol",
                    "Column that correspends to CoNLLU(formCol=) output",
                    typeConverter=TypeConverters.toString)

    lemmaCol = Param(Params._dummy(),
                     "lemmaCol",
                     "Column that correspends to CoNLLU(lemmaCol=) output",
                     typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(Lemmatizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Lemmatizer")
        self._setDefault(
            formCol="form",
            lemmaCol="lemma"
        )

    def _create_model(self, java_model):
        return LemmatizerModel(java_model=java_model)

    def setFormCol(self, value):
        """Column that correspends to CoNLLU(formCol=) output

        Parameters
        ----------
        value : str
            Name of column for Array of Form tokens
        """
        return self._set(formCol=value)

    def setLemmaCol(self, value):
        """Column that correspends to CoNLLU(fromLemma=) output

        Parameters
        ----------
        value : str
            Name of column for Array of Lemma tokens
        """
        return self._set(lemmaCol=value)

    def setDictionary(self, path, key_delimiter, value_delimiter, read_as=ReadAs.TEXT,
                      options={"format": "text"}):
        """Sets the external dictionary for the lemmatizer.

        Parameters
        ----------
        path : str
            Path to the source files
        key_delimiter : str
            Delimiter for the key
        value_delimiter : str
            Delimiter for the values
        read_as : str, optional
            How to read the file, by default ReadAs.TEXT
        options : dict, optional
            Options to read the resource, by default {"format": "text"}

        Examples
        --------
        Here the file has each key is delimited by ``"->"`` and values are
        delimited by ``\\t``::

            ...
            pick	->	pick	picks	picking	picked
            peck	->	peck	pecking	pecked	pecks
            pickle	->	pickle	pickles	pickled	pickling
            pepper	->	pepper	peppers	peppered	peppering
            ...

        This file can then be parsed with

        >>> lemmatizer = Lemmatizer() \\
        ...     .setInputCols(["token"]) \\
        ...     .setOutputCol("lemma") \\
        ...     .setDictionary("lemmas_small.txt", "->", "\\t")
        """
        opts = options.copy()
        if "keyDelimiter" not in opts:
            opts["keyDelimiter"] = key_delimiter
        if "valueDelimiter" not in opts:
            opts["valueDelimiter"] = value_delimiter
        return self._set(dictionary=ExternalResource(path, read_as, opts))


class LemmatizerModel(AnnotatorModel):
    """Instantiated Model of the Lemmatizer.

    This is the instantiated model of the :class:`.Lemmatizer`.
    For training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> lemmatizer = LemmatizerModel.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("lemma")

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models?task=Lemmatization>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    None

    Examples
    --------
    The lemmatizer from the example of the :class:`.Lemmatizer` can be replaced
    with:

    >>> lemmatizer = LemmatizerModel.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("lemma")
    """
    name = "LemmatizerModel"

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.LemmatizerModel", java_model=None):
        super(LemmatizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="lemma_antbnc", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "lemma_antbnc"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        LemmatizerModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(LemmatizerModel, name, lang, remote_loc)
