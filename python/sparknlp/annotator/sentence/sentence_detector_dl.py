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
"""Contains classes for SentenceDetectorDl."""

from sparknlp.common import *


class SentenceDetectorDLApproach(AnnotatorApproach):
    """Trains an annotator that detects sentence boundaries using a deep
    learning approach.

    Currently, only the CNN model is supported for training, but in the future
    the architecture of the model can be set with :meth:`.setModel`.

    For pretrained models see :class:`.SentenceDetectorDLModel`.

    Each extracted sentence can be returned in an Array or exploded to separate
    rows, if ``explodeSentences`` is set to ``True``.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    modelArchitecture
        Model architecture (CNN)
    impossiblePenultimates
        Impossible penultimates - list of strings which a sentence can't end
        with
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each
    epochsNumber
        Number of epochs for the optimization process
    outputLogsPath
        Path to folder where logs will be saved. If no path is specified, no
        logs are generated
    explodeSentences
        Whether to explode each sentence into a different row, for better
        parallelization. Defaults to False.

    References
    ----------
    The default model ``"cnn"`` is based on the paper `Deep-EOS: General-Purpose
    Neural Networks for Sentence Boundary Detection (2020, Stefan Schweter,
    Sajawel Ahmed)
    <https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_41.pdf>`__
    using a CNN architecture. We also modified the original implementation a
    little bit to cover broken sentences and some impossible end of line chars.

    Examples
    --------
    The training process needs data, where each data point is a sentence.
    In this example the ``train.txt`` file has the form of::

        ...
        Slightly more moderate language would make our present situation – namely the lack of progress – a little easier.
        His political successors now have great responsibilities to history and to the heritage of values bequeathed to them by Nelson Mandela.
        ...

    where each line is one sentence.

    Training can then be started like so:

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> trainingData = spark.read.text("train.txt").toDF("text")
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentenceDetector = SentenceDetectorDLApproach() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentences") \\
    ...     .setEpochsNumber(100)
    >>> pipeline = Pipeline().setStages([documentAssembler, sentenceDetector])
    >>> model = pipeline.fit(trainingData)
    """

    name = "SentenceDetectorDLApproach"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    modelArchitecture = Param(Params._dummy(),
                              "modelArchitecture",
                              "Model architecture (CNN)",
                              typeConverter=TypeConverters.toString)

    impossiblePenultimates = Param(Params._dummy(),
                                   "impossiblePenultimates",
                                   "Impossible penultimates - list of strings which a sentence can't end with",
                                   typeConverter=TypeConverters.toListString)

    validationSplit = Param(Params._dummy(),
                            "validationSplit",
                            "Choose the proportion of training dataset to be validated against the model on each "
                            "Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    epochsNumber = Param(Params._dummy(),
                         "epochsNumber",
                         "Number of epochs for the optimization process",
                         TypeConverters.toInt)

    outputLogsPath = Param(Params._dummy(),
                           "outputLogsPath",
                           "Path to folder where logs will be saved. If no path is specified, no logs are generated",
                           TypeConverters.toString)

    explodeSentences = Param(Params._dummy(),
                             "explodeSentences",
                             "whether to explode each sentence into a different row, for better parallelization. Defaults to false.",
                             TypeConverters.toBoolean)

    def setModel(self, model_architecture):
        """Sets the Model architecture. Currently only ``"cnn"`` is available.

        Parameters
        ----------
        model_architecture : str
            Model architecture
        """
        return self._set(modelArchitecture=model_architecture)

    def setValidationSplit(self, validation_split):
        """Sets the proportion of training dataset to be validated against the
        model on each Epoch, by default it is 0.0 and off. The value should be
        between 0.0 and 1.0.

        Parameters
        ----------
        validation_split : float
            Proportion of training dataset to be validated
        """
        return self._set(validationSplit=validation_split)

    def setEpochsNumber(self, epochs_number):
        """Sets number of epochs to train.

        Parameters
        ----------
        epochs_number : int
            Number of epochs
        """
        return self._set(epochsNumber=epochs_number)

    def setOutputLogsPath(self, output_logs_path):
        """Sets folder path to save training logs.

        Parameters
        ----------
        output_logs_path : str
            Folder path to save training logs
        """
        return self._set(outputLogsPath=output_logs_path)

    def setImpossiblePenultimates(self, impossible_penultimates):
        """Sets impossible penultimates - list of strings which a sentence can't
        end with.

        Parameters
        ----------
        impossible_penultimates : List[str]
            List of strings which a sentence can't end with

        """
        return self._set(impossiblePenultimates=impossible_penultimates)

    def setExplodeSentences(self, value):
        """Sets whether to explode each sentence into a different row, for
        better parallelization, by default False.

        Parameters
        ----------
        value : bool
            Whether to explode each sentence into a different row
        """
        return self._set(explodeSentences=value)

    def _create_model(self, java_model):
        return SentenceDetectorDLModel(java_model=java_model)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach"):
        super(SentenceDetectorDLApproach, self).__init__(classname=classname)


class SentenceDetectorDLModel(AnnotatorModel, HasEngine):
    """Annotator that detects sentence boundaries using a deep learning approach.

    Instantiated Model of the :class:`.SentenceDetectorDLApproach`.
    Detects sentence boundaries using a deep learning approach.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> sentenceDL = SentenceDetectorDLModel.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentencesDL")

    The default model is ``"sentence_detector_dl"``, if no name is provided. For
    available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Sentence+Detection>`__.

    Each extracted sentence can be returned in an Array or exploded to separate
    rows, if ``explodeSentences`` is set to ``true``.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/multilingual/SentenceDetectorDL.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    modelArchitecture
        Model architecture (CNN)
    explodeSentences
        whether to explode each sentence into a different row, for better
        parallelization. Defaults to false.
    customBounds
        characters used to explicitly mark sentence bounds, by default []
    useCustomBoundsOnly
        Only utilize custom bounds in sentence detection, by default False
    splitLength
        length at which sentences will be forcibly split
    minLength
        Set the minimum allowed length for each sentence, by default 0
    maxLength
        Set the maximum allowed length for each sentence, by default 99999
    impossiblePenultimates
        Impossible penultimates - list of strings which a sentence can't end
        with

    Examples
    --------
    In this example, the normal `SentenceDetector` is compared to the
    `SentenceDetectorDLModel`. In a pipeline, `SentenceDetectorDLModel` can be
    used as a replacement for the `SentenceDetector`.

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentences")
    >>> sentenceDL = SentenceDetectorDLModel \\
    ...     .pretrained("sentence_detector_dl", "en") \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentencesDL")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     sentenceDL
    ... ])
    >>> data = spark.createDataFrame([[\"\"\"John loves Mary.Mary loves Peter
    ...     Peter loves Helen .Helen loves John;
    ...     Total: four people involved.\"\"\"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(sentences.result) as sentences").show(truncate=False)
    +----------------------------------------------------------+
    |sentences                                                 |
    +----------------------------------------------------------+
    |John loves Mary.Mary loves Peter\\n     Peter loves Helen .|
    |Helen loves John;                                         |
    |Total: four people involved.                              |
    +----------------------------------------------------------+
    >>> result.selectExpr("explode(sentencesDL.result) as sentencesDL").show(truncate=False)
    +----------------------------+
    |sentencesDL                 |
    +----------------------------+
    |John loves Mary.            |
    |Mary loves Peter            |
    |Peter loves Helen .         |
    |Helen loves John;           |
    |Total: four people involved.|
    +----------------------------+
    """
    name = "SentenceDetectorDLModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    modelArchitecture = Param(Params._dummy(), "modelArchitecture", "Model architecture (CNN)",
                              typeConverter=TypeConverters.toString)

    explodeSentences = Param(Params._dummy(),
                             "explodeSentences",
                             "whether to explode each sentence into a different row, for better parallelization. Defaults to false.",
                             TypeConverters.toBoolean)

    customBounds = Param(Params._dummy(),
                         "customBounds",
                         "characters used to explicitly mark sentence bounds",
                         typeConverter=TypeConverters.toListString)

    useCustomBoundsOnly = Param(Params._dummy(),
                                "useCustomBoundsOnly",
                                "Only utilize custom bounds in sentence detection",
                                typeConverter=TypeConverters.toBoolean)

    splitLength = Param(Params._dummy(),
                        "splitLength",
                        "length at which sentences will be forcibly split.",
                        typeConverter=TypeConverters.toInt)

    minLength = Param(Params._dummy(),
                      "minLength",
                      "Set the minimum allowed length for each sentence.",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "Set the maximum allowed length for each sentence",
                      typeConverter=TypeConverters.toInt)

    impossiblePenultimates = Param(Params._dummy(),
                                   "impossiblePenultimates",
                                   "Impossible penultimates - list of strings which a sentence can't end with",
                                   typeConverter=TypeConverters.toListString)

    def setModel(self, modelArchitecture):
        """Sets the Model architecture. Currently only ``"cnn"`` is available.

        Parameters
        ----------
        model_architecture : str
            Model architecture
        """
        return self._set(modelArchitecture=modelArchitecture)

    def setExplodeSentences(self, value):
        """Sets whether to explode each sentence into a different row, for
        better parallelization, by default False.

        Parameters
        ----------
        value : bool
            Whether to explode each sentence into a different row
        """
        return self._set(explodeSentences=value)

    def setCustomBounds(self, value):
        """Sets characters used to explicitly mark sentence bounds, by default
        [].

        Parameters
        ----------
        value : List[str]
            Characters used to explicitly mark sentence bounds
        """
        return self._set(customBounds=value)

    def setUseCustomBoundsOnly(self, value):
        """Sets whether to only utilize custom bounds in sentence detection, by
        default False.

        Parameters
        ----------
        value : bool
            Whether to only utilize custom bounds
        """
        return self._set(useCustomBoundsOnly=value)

    def setSplitLength(self, value):
        """Sets length at which sentences will be forcibly split.

        Parameters
        ----------
        value : int
            Length at which sentences will be forcibly split.
        """
        return self._set(splitLength=value)

    def setMinLength(self, value):
        """Sets minimum allowed length for each sentence, by default 0

        Parameters
        ----------
        value : int
            Minimum allowed length for each sentence
        """
        return self._set(minLength=value)

    def setMaxLength(self, value):
        """Sets the maximum allowed length for each sentence, by default
        99999

        Parameters
        ----------
        value : int
            Maximum allowed length for each sentence
        """
        return self._set(maxLength=value)

    def setImpossiblePenultimates(self, impossible_penultimates):
        """Sets impossible penultimates - list of strings which a sentence can't
        end with.

        Parameters
        ----------
        impossible_penultimates : List[str]
            List of strings which a sentence can't end with

        """
        return self._set(impossiblePenultimates=impossible_penultimates)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel",
                 java_model=None):
        super(SentenceDetectorDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            useCustomBoundsOnly=False,
            customBounds=[],
            explodeSentences=False,
            minLength=0,
            maxLength=99999
        )

    @staticmethod
    def pretrained(name="sentence_detector_dl", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sentence_detector_dl"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        SentenceDetectorDLModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SentenceDetectorDLModel, name, lang, remote_loc)
