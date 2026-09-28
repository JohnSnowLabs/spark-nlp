#  Copyright 2017-2026 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Contains classes concerning DolphinForDocumentParsing."""

from sparknlp.common import *


class DolphinForDocumentParsing(AnnotatorModel,
                                HasBatchedAnnotateImage,
                                HasImageFeatureProperties,
                                HasRescaleFactor,
                                HasEngine):
    """Document image parsing with ByteDance Dolphin.

    Dolphin follows an analyze-then-parse design: stage 1 generates the page's layout
    elements in reading order, then each element is cropped and parsed independently in
    stage 2, batched by type. Tables come back as HTML, formulas as LaTeX, text as markdown.

    Dolphin 1.5 and 1.0 load the same way. The release is recognised from its stage-1 output,
    which decides the box geometry and whether formulas and code get their own prompts
    (1.5 only).

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object:

    >>> dolphin = DolphinForDocumentParsing.pretrained() \\
    ...     .setInputCols(["image_assembler"]) \\
    ...     .setOutputCol("elements")

    The default model is ``"dolphin_1_5"``.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    parsingMode
        What the input image is. ``page`` (default) runs both stages on a whole page,
        ``layout`` runs stage 1 only, and ``table``, ``text``, ``formula`` or ``code`` run
        stage 2 on an already cropped region. ``formula`` and ``code`` are Dolphin 1.5
        prompts.
    outputFormat
        ``elements`` (default) emits one annotation per document element carrying
        ``dolphinLabel``, ``bbox``, ``readingOrder`` and ``elementType`` metadata;
        ``markdown`` emits a single assembled document per page.
    elementBatchSize
        Element crops decoded per forward pass, default 4.
    maxOutputLength
        Maximum tokens generated per element, default 4096. Generation always stops at the
        decoder's 4096-position window, prompt included.
    layoutMaxOutputLength
        Maximum tokens generated for the layout pass, default 4096.
    repetitionPenalty
        Repetition penalty, default 1.0 (off) as in Dolphin 1.5's reference pipeline;
        Dolphin 1.0's used 1.1.
    customPrompt
        Overrides the prompt in the ``table``, ``text``, ``formula`` and ``code`` modes.
    adjustBoxEdges
        Dolphin 1.0 only: widen its boxes so their edges do not cut through ink, default
        True. 1.5 boxes are used as generated.
    minElementSize
        Page mode drops elements whose crop width or height is this many pixels or fewer,
        default 3, as the reference does.

    Notes
    -----
    The task comes from ``parsingMode``, not from the image annotation's ``text`` field.
    ``Reader2Image`` defaults its ``promptTemplate`` to ``qwen2vl-chat``, and that markup
    would otherwise be fed to Dolphin as a task prompt.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> imageDF = spark.read.format("image").load(path="pages/")
    >>> imageAssembler = ImageAssembler() \\
    ...     .setInputCol("image") \\
    ...     .setOutputCol("image_assembler")
    >>> dolphin = DolphinForDocumentParsing.pretrained() \\
    ...     .setInputCols("image_assembler") \\
    ...     .setOutputCol("elements") \\
    ...     .setParsingMode("page")
    >>> pipeline = Pipeline().setStages([imageAssembler, dolphin])
    >>> result = pipeline.fit(imageDF).transform(imageDF)
    >>> result.selectExpr("explode(elements) as e") \\
    ...     .selectExpr("e.metadata['dolphinLabel'] as label", "e.result") \\
    ...     .show(truncate=60)
    """

    name = "DolphinForDocumentParsing"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    parsingMode = Param(Params._dummy(), "parsingMode",
                        "One of: page, layout, table, text, formula, code",
                        typeConverter=TypeConverters.toString)

    outputFormat = Param(Params._dummy(), "outputFormat",
                         "One of: elements, markdown",
                         typeConverter=TypeConverters.toString)

    elementBatchSize = Param(Params._dummy(), "elementBatchSize",
                             "Element crops decoded per forward pass",
                             typeConverter=TypeConverters.toInt)

    maxOutputLength = Param(Params._dummy(), "maxOutputLength",
                            "Maximum tokens generated per element",
                            typeConverter=TypeConverters.toInt)

    layoutMaxOutputLength = Param(Params._dummy(), "layoutMaxOutputLength",
                                  "Maximum tokens generated for the layout pass",
                                  typeConverter=TypeConverters.toInt)

    repetitionPenalty = Param(Params._dummy(), "repetitionPenalty",
                              "Repetition penalty; 1.0 disables it",
                              typeConverter=TypeConverters.toFloat)

    customPrompt = Param(Params._dummy(), "customPrompt",
                         "Overrides the prompt in the table, text, formula and code modes",
                         typeConverter=TypeConverters.toString)

    adjustBoxEdges = Param(Params._dummy(), "adjustBoxEdges",
                           "Widen Dolphin 1.0 boxes so edges do not cut through ink",
                           typeConverter=TypeConverters.toBoolean)

    minElementSize = Param(Params._dummy(), "minElementSize",
                           "Drop elements whose crop width or height is this many pixels or fewer",
                           typeConverter=TypeConverters.toInt)

    #: Modes accepted by ``setParsingMode``; mirrors DolphinForDocumentParsing.ParsingModes.
    PARSING_MODES = ["page", "layout", "table", "text", "formula", "code"]

    #: Formats accepted by ``setOutputFormat``; mirrors DolphinForDocumentParsing.OutputFormats.
    OUTPUT_FORMATS = ["elements", "markdown"]

    def setParsingMode(self, value):
        """Sets the parsing mode: page, layout, table, text, formula or code."""
        if value not in self.PARSING_MODES:
            raise ValueError(
                "parsingMode must be one of %s" % ", ".join(self.PARSING_MODES))
        return self._set(parsingMode=value)

    def setOutputFormat(self, value):
        """Sets the output format: elements or markdown."""
        if value not in self.OUTPUT_FORMATS:
            raise ValueError(
                "outputFormat must be one of %s" % ", ".join(self.OUTPUT_FORMATS))
        return self._set(outputFormat=value)

    def setElementBatchSize(self, value):
        """Sets how many element crops are decoded per forward pass."""
        if value <= 0:
            raise ValueError("elementBatchSize must be positive.")
        return self._set(elementBatchSize=value)

    def setMaxOutputLength(self, value):
        """Sets the maximum tokens generated per element."""
        if value <= 0:
            raise ValueError("maxOutputLength must be positive.")
        return self._set(maxOutputLength=value)

    def setLayoutMaxOutputLength(self, value):
        """Sets the maximum tokens generated for the layout pass."""
        return self._set(layoutMaxOutputLength=value)

    def setRepetitionPenalty(self, value):
        """Sets the repetition penalty."""
        return self._set(repetitionPenalty=value)

    def setCustomPrompt(self, value):
        """Sets a prompt that overrides the default in the single-crop modes."""
        return self._set(customPrompt=value)

    def setAdjustBoxEdges(self, value):
        """Sets whether Dolphin 1.0 boxes are widened away from ink."""
        return self._set(adjustBoxEdges=value)

    def setMinElementSize(self, value):
        """Sets the crop size at or below which page mode drops an element."""
        return self._set(minElementSize=value)

    def getParsingMode(self):
        """Gets the parsing mode."""
        return self.getOrDefault(self.parsingMode)

    def getOutputFormat(self):
        """Gets the output format: elements or markdown."""
        return self.getOrDefault(self.outputFormat)

    def getElementBatchSize(self):
        """Gets how many element crops are decoded per forward pass."""
        return self.getOrDefault(self.elementBatchSize)

    def getMaxOutputLength(self):
        """Gets the maximum tokens generated per element."""
        return self.getOrDefault(self.maxOutputLength)

    def getLayoutMaxOutputLength(self):
        """Gets the maximum tokens generated for the layout pass."""
        return self.getOrDefault(self.layoutMaxOutputLength)

    def getRepetitionPenalty(self):
        """Gets the repetition penalty."""
        return self.getOrDefault(self.repetitionPenalty)

    def getCustomPrompt(self):
        """Gets the prompt overriding the default in the single-crop modes."""
        return self.getOrDefault(self.customPrompt)

    def getAdjustBoxEdges(self):
        """Gets whether Dolphin 1.0 boxes are widened away from ink."""
        return self.getOrDefault(self.adjustBoxEdges)

    def getMinElementSize(self):
        """Gets the crop size at or below which page mode drops an element."""
        return self.getOrDefault(self.minElementSize)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.DolphinForDocumentParsing",
                 java_model=None):
        super(DolphinForDocumentParsing, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=1,
            parsingMode="page",
            outputFormat="elements",
            elementBatchSize=4,
            maxOutputLength=4096,
            layoutMaxOutputLength=4096,
            repetitionPenalty=1.0,
            customPrompt="",
            adjustBoxEdges=True,
            minElementSize=3,
            size=896,
            doNormalize=True,
            doResize=True,
            doRescale=True,
            rescaleFactor=1 / 255.0,
            resample=2,
            imageMean=[0.485, 0.456, 0.406],
            imageStd=[0.229, 0.224, 0.225],
            featureExtractorType="DonutFeatureExtractor"
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        DolphinForDocumentParsing
            The restored model
        """
        from sparknlp.internal import _DolphinForDocumentParsingLoader
        jModel = _DolphinForDocumentParsingLoader(folder, spark_session._jsparkSession)._java_obj
        return DolphinForDocumentParsing(java_model=jModel)

    @staticmethod
    def pretrained(name="dolphin_1_5", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "dolphin_1_5"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        DolphinForDocumentParsing
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(DolphinForDocumentParsing, name, lang, remote_loc)
