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
"""Contains classes for the LLMNerModel."""

from sparknlp.common import *


class LLMNerModel(AnnotatorModel, HasBatchedAnnotate, HasLlamaCppProperties):
    """End-to-end LLM-based Named Entity Recognition using AutoGGUF with BNF grammars.

    LLMNerModel is an end-to-end annotator that performs entity extraction from text
    using Large Language Models (LLMs) with structured JSON output via BNF grammars.
    It embeds AutoGGUFModel directly and uses string matching to compute accurate
    character indices for extracted entities.

    This annotator follows the LangExtract pattern from Google Research, combining
    few-shot prompting with constrained generation through llama.cpp BNF grammars to
    ensure valid JSON output.

    The LLM generates responses in this format (enforced by grammar)::

        {
          "extractions": [
            {"entity": "MEDICATION", "text": "aspirin"},
            {"entity": "DOSAGE", "text": "250mg"}
          ]
        }

    The annotator performs string matching to find the exact character positions
    of each entity in the original text, outputting CHUNK annotations with accurate
    begin/end indices and chunk indexing similar to other Spark NLP annotators.

    The model is instantiated directly and automatically loads the specified AutoGGUF
    model at runtime:

    >>> llm_ner = LLMNerModel() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("entities") \\
    ...     .setModelName("qwen3_4b_bf16_gguf") \\
    ...     .setEntityTypes(["PERSON", "ORGANIZATION", "LOCATION"])

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    promptTemplate : str, optional
        Custom prompt template for NER extraction. Use {entityTypes} placeholder.
    entityTypes : List[str], optional
        List of entity types to extract (used in prompt), by default
        ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME"]
    caseSensitive : bool, optional
        Whether entity matching is case-sensitive, by default False
    modelName : str, optional
        Name of the AutoGGUF model to load, by default "qwen3_4b_bf16_gguf"
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
    >>> llmNer = LLMNerModel() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("entities") \\
    ...     .setModelName("qwen3_4b_bf16_gguf") \\
    ...     .setEntityTypes(["MEDICATION", "DOSAGE", "ROUTE", "FREQUENCY"]) \\
    ...     .setNPredict(500) \\
    ...     .setTemperature(0.1)
    >>> pipeline = Pipeline().setStages([documentAssembler, llmNer])
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

    name = "LLMNerModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.CHUNK

    promptTemplate = Param(
        Params._dummy(),
        "promptTemplate",
        "Custom prompt template for NER extraction. Use {entityTypes} placeholder.",
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

    modelName = Param(
        Params._dummy(),
        "modelName",
        "Name of the AutoGGUF model to load for NER extraction",
        typeConverter=TypeConverters.toString,
    )

    fewShotExamples = Param(
        Params._dummy(),
        "fewShotExamples",
        "Few-shot examples as array of (input, output_json) tuples",
    )

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.dl.LLMNerModel", java_model=None):
        super(LLMNerModel, self).__init__(
            classname=classname,
            java_model=java_model,
        )
        self._setDefault(
            entityTypes=["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME"],
            caseSensitive=False,
            modelName="qwen3_4b_bf16_gguf",
            useChatTemplate=True,
            nCtx=4096,
            nBatch=512,
            nPredict=500,
            nGpuLayers=99,
            batchSize=4
        )

    def setPromptTemplate(self, value):
        """Set custom prompt template for NER extraction.

        Parameters
        ----------
        value : str
            Custom prompt template. Use {entityTypes} and {text} as placeholders.

        Returns
        -------
        LLMNerModel
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
        LLMNerModel
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
        LLMNerModel
            The updated model
        """
        return self._set(caseSensitive=value)

    def setModelName(self, value):
        """Set the name of the AutoGGUF model to load.

        Parameters
        ----------
        value : str
            Name of the pretrained AutoGGUF model (e.g., "qwen3_4b_bf16_gguf")

        Returns
        -------
        LLMNerModel
            The updated model
        """
        return self._set(modelName=value)

    def setFewShotExamples(self, value):
        """Set few-shot examples to guide the model.

        Parameters
        ----------
        value : List[Tuple[str, str]]
            List of (input_text, json_output) tuples as examples

        Returns
        -------
        LLMNerModel
            The updated model
        """
        return self._set(fewShotExamples=value)

    def getPromptTemplate(self):
        """Get the custom prompt template for NER extraction.

        Returns
        -------
        str
            The prompt template
        """
        return self.getOrDefault(self.promptTemplate)

    def getEntityTypes(self):
        """Get the list of entity types to extract.

        Returns
        -------
        List[str]
            List of entity type names
        """
        return self.getOrDefault(self.entityTypes)


    def getCaseSensitive(self):
        """Get whether entity matching is case-sensitive.

        Returns
        -------
        bool
            True if case-sensitive, False otherwise
        """
        return self.getOrDefault(self.caseSensitive)

    def getModelName(self):
        """Get the name of the AutoGGUF model.

        Returns
        -------
        str
            Name of the pretrained AutoGGUF model
        """
        return self.getOrDefault(self.modelName)

    def getFewShotExamples(self):
        """Get the few-shot examples.

        Returns
        -------
        List[Tuple[str, str]]
            List of (input_text, json_output) tuples as examples
        """
        return self.getOrDefault(self.fewShotExamples)

