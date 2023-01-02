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

"""Contains classes concerning HubertForCTC."""

from sparknlp.common import *


class HubertForCTC(AnnotatorModel,
                   HasBatchedAnnotateAudio,
                   HasAudioFeatureProperties,
                   HasEngine):
    """Hubert Model with a language modeling head on top for Connectionist Temporal
    Classification (CTC). Hubert was proposed in HuBERT: Self-Supervised Speech
    Representation Learning by Masked Prediction of Hidden Units by Wei-Ning Hsu,
    Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov,
    Abdelrahman Mohamed.

    The annotator takes audio files and transcribes it as text. The audio needs to be
    provided pre-processed an array of floats.

    Note that this annotator is currently not supported on Apple Silicon processors such
    as the M1. This is due to the processor not supporting instructions for XLA.

    Pretrained models can be loaded with ``pretrained`` of the companion object:

    >>> speechToText = HubertForCTC.pretrained() \\
    ...     .setInputCols(["audio_assembler"]) \\
    ...     .setOutputCol("text")


    The default model is ``"asr_hubert_large_ls960"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://nlp.johnsnowlabs.com/models>`__.

    To see which models are compatible and how to import them see
    https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
    examples, see
    `HubertForCTCTestSpec <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/audio/HubertForCTCTestSpec.scala>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``AUDIO``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------

    batchSize
        Size of each batch, by default 4

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> audioAssembler = AudioAssembler() \\
    ...     .setInputCol("audio_content") \\
    ...     .setOutputCol("audio_assembler")
    >>> speechToText = HubertForCTC \\
    ...     .pretrained() \\
    ...     .setInputCols(["audio_assembler"]) \\
    ...     .setOutputCol("text")
    >>> pipeline = Pipeline().setStages([audioAssembler, speechToText])
    >>> processedAudioFloats = spark.createDataFrame([[rawFloats]]).toDF("audio_content")
    >>> result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
    >>> result.select("text.result").show(truncate = False)
    +------------------------------------------------------------------------------------------+
    |result                                                                                    |
    +------------------------------------------------------------------------------------------+
    |[MISTER QUILTER IS THE APOSTLE OF THE MIDLE CLASES AND WE ARE GLAD TO WELCOME HIS GOSPEL ]|
    +------------------------------------------------------------------------------------------+
    """
    name = "HubertForCTC"

    inputAnnotatorTypes = [AnnotatorType.AUDIO]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with "
                             "config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.audio.HubertForCTC",
                 java_model=None):
        super(HubertForCTC, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=4
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
        HubertForCTC
            The restored model
        """
        from sparknlp.internal import _HubertForCTC
        jModel = _HubertForCTC(folder, spark_session._jsparkSession)._java_obj
        return HubertForCTC(java_model=jModel)

    @staticmethod
    def pretrained(name="asr_hubert_large_ls960", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "asr_hubert_large_ls960"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        HubertForCTC
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(HubertForCTC, name, lang, remote_loc)
