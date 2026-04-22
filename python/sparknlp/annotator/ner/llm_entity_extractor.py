#  Copyright 2017-2024 John Snow Labs
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
"""Contains classes for the LLMEntityExtractor annotator."""

from sparknlp.common import *


class LLMEntityExtractor(AnnotatorModel, HasBatchedAnnotate, HasLlamaCppProperties):
    """End-to-end LLM-based entity extraction using AutoGGUF with BNF grammars.

    LLMEntityExtractor is an end-to-end annotator that performs entity extraction
    from text using Large Language Models (LLMs) with structured JSON output via
    BNF grammars. It embeds AutoGGUFModel directly and uses string matching to
    compute accurate character indices for extracted entities.

    This annotator follows the LangExtract pattern from Google Research,
    combining few-shot prompting with constrained generation through llama.cpp
    BNF grammars to ensure valid JSON output.

    The LLM generates responses in this format (enforced by grammar)::

        {
          "extractions": [
            {"entity": "MEDICATION", "text": "aspirin"},
            {"entity": "DOSAGE", "text": "250mg"}
          ]
        }

    The annotator performs string matching to find the exact character positions
    of each entity in the original text, outputting CHUNK annotations with
    accurate begin/end indices and chunk indexing similar to other Spark NLP
    annotators.

    The model is loaded via ``LLMEntityExtractor.pretrained()`` to download a
    pretrained model, or ``LLMEntityExtractor.loadSavedModel()`` to load a local
    GGUF model:

    >>> entity_extractor = LLMEntityExtractor.pretrained("qwen3_4b_bf16_gguf") \
    ...     .setInputCols(["document"]) \
    ...     .setOutputCol("entities") \
    ...     .setEntityTypes(["PERSON", "ORGANIZATION", "LOCATION"])

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    promptTemplate : str, optional
        Custom prompt template for entity extraction. Use {entityTypes}
        placeholder.
    entityTypes : List[str], optional
        List of entity types to extract (used in prompt), by default
        ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME"]
    caseSensitive : bool, optional
        Whether entity matching is case-sensitive, by default False
    fewShotExamples : List[Tuple[str, str]], optional
        Few-shot examples as (input, output_json) tuples to guide the model

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> entity_extractor = LLMEntityExtractor.pretrained("qwen3_4b_bf16_gguf") \
    ...     .setInputCols(["document"]) \
    ...     .setOutputCol("entities") \
    ...     .setEntityTypes(["MEDICATION", "DOSAGE", "ROUTE", "FREQUENCY"]) \
    ...     .setNPredict(500) \\
    ...     .setTemperature(0.1)
    >>> pipeline = Pipeline().setStages([documentAssembler, entity_extractor])
    >>> data = spark.createDataFrame([["Patient prescribed 500mg amoxicillin PO TID"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("entities.result", "entities.metadata").show(truncate=False)
    +------------------------------+--------------------------------+
    |result                        |metadata                        |
    +------------------------------+--------------------------------+
    |[500mg, amoxicillin, PO, TID] |[{entity -> DOSAGE}, ...]       |
    +------------------------------+--------------------------------+

    See Also
    --------
    NerDLModel : for traditional BiLSTM-CRF NER
    NerConverter : to further process NER results
    """

    name = "LLMEntityExtractor"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.CHUNK

    promptTemplate = Param(
        Params._dummy(),
        "promptTemplate",
        "Custom prompt template for entity extraction. Use {entityTypes} placeholder.",
        typeConverter=TypeConverters.toString,
    )

    entityTypes = Param(
        Params._dummy(),
        "entityTypes",
        "List of entity types to extract (used in prompt)",
        typeConverter=TypeConverters.toListString,
    )

    caseSensitive = Param(
        Params._dummy(),
        "caseSensitive",
        "Whether entity matching is case-sensitive",
        typeConverter=TypeConverters.toBoolean,
    )

    fewShotExamples = Param(
        Params._dummy(),
        "fewShotExamples",
        "Few-shot examples as array of (input, output_json) tuples",
    )

    def __init__(
        self,
        classname="com.johnsnowlabs.nlp.annotators.ner.dl.LLMEntityExtractor",
        java_model=None,
    ):
        super(LLMEntityExtractor, self).__init__(
            classname=classname,
            java_model=java_model,
        )
        self._setDefault(
            entityTypes=["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME"],
            caseSensitive=False,
            useChatTemplate=True,
            nCtx=4096,
            nBatch=512,
            nPredict=500,
            nGpuLayers=99,
            batchSize=4,
        )

    def setPromptTemplate(self, value):
        """Set custom prompt template for entity extraction.

        Parameters
        ----------
        value : str
            Custom prompt template. Use {entityTypes} and {text} as placeholders.

        Returns
        -------
        LLMEntityExtractor
            The updated model
        """
        return self._set(promptTemplate=value)

    def setEntityTypes(self, value):
        """Set the list of entity types to extract.

        Parameters
        ----------
        value : List[str]
            List of entity type names

        Returns
        -------
        LLMEntityExtractor
            The updated model
        """
        return self._set(entityTypes=value)

    def setCaseSensitive(self, value):
        """Set whether entity matching is case-sensitive.

        Parameters
        ----------
        value : bool
            True for case-sensitive matching, False for case-insensitive

        Returns
        -------
        LLMEntityExtractor
            The updated model
        """
        return self._set(caseSensitive=value)

    def setFewShotExamples(self, value):
        """Set few-shot examples to guide the model.

        Parameters
        ----------
        value : List[Tuple[str, str]]
            List of (input_text, json_output) tuples as examples

        Returns
        -------
        LLMEntityExtractor
            The updated model
        """
        java_compatible = [list(pair) for pair in value]
        self._call_java("setFewShotExamples", java_compatible)
        return self

    def getPromptTemplate(self):
        """Get the custom prompt template for entity extraction."""
        return self.getOrDefault(self.promptTemplate)

    def getEntityTypes(self):
        """Get the list of entity types to extract."""
        return self.getOrDefault(self.entityTypes)

    def getCaseSensitive(self):
        """Get whether entity matching is case-sensitive."""
        return self.getOrDefault(self.caseSensitive)

    def getFewShotExamples(self):
        """Get the few-shot examples."""
        return self.getOrDefault(self.fewShotExamples)

    @classmethod
    def loadSavedModel(cls, path, spark_session):
        """Loads a locally saved GGUF model for LLM-based entity extraction."""
        from sparknlp.internal import _LLMEntityExtractorLoader

        jModel = _LLMEntityExtractorLoader(path, spark_session._jsparkSession)._java_obj
        return cls(java_model=jModel)

    @classmethod
    def pretrained(cls, name="qwen3_4b_bf16_gguf", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model."""
        from sparknlp.pretrained import ResourceDownloader

        return ResourceDownloader.downloadModel(cls, name, lang, remote_loc)

    def close(self):
        """Closes the underlying llama.cpp model backend freeing resources."""
        self._java_obj.close()
