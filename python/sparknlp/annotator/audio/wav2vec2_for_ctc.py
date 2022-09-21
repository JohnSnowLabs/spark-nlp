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

"""Contains classes concerning Wav2Vec2ForCTC."""

from sparknlp.common import *


class Wav2Vec2ForCTC(AnnotatorModel,
                     HasBatchedAnnotateAudio,
                     HasAudioFeatureProperties):
    """

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``AUDIO``              ``DOCUMENT``
    ====================== ======================

    References
    ----------


    Parameters
    ----------

    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> audioAssembler = AudioAssembler() \\
    ...     .setInputCol("audio") \\
    ...     .setOutputCol("audio_assembler")
    >>> speechToText = Wav2Vec2ForCTC \\
    ...     .pretrained() \\
    ...     .setInputCols(["audio_assembler"]) \\
    ...     .setOutputCol("text")
    >>> pipeline = Pipeline().setStages([audioAssembler, Wav2Vec2ForCTC])
    >>> pipelineDF = pipeline.fit(audioDF).transform(audioDF)

    """
    name = "Wav2Vec2ForCTC"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.audio.Wav2Vec2ForCTC",
                 java_model=None):
        super(Wav2Vec2ForCTC, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=2
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
        Wav2Vec2ForCTC
            The restored model
        """
        from sparknlp.internal import _Wav2Vec2ForCTC
        jModel = _Wav2Vec2ForCTC(folder, spark_session._jsparkSession)._java_obj
        return Wav2Vec2ForCTC(java_model=jModel)

    @staticmethod
    def pretrained(name="wav2vec2_base_960h", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "wav2vec2_base_960h"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        Wav2Vec2ForCTC
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(Wav2Vec2ForCTC, name, lang, remote_loc)
