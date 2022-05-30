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

from sparknlp.common import *


class SpanBertCorefModel(AnnotatorModel,
                     HasEmbeddingsProperties,
                     HasCaseSensitiveProperties,
                     HasStorageRef):
    """
    TODO
    """

    name = "SpanBertCorefModel"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.coref.SpanBertCorefModel", java_model=None):
        super(SpanBertCorefModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            maxSentenceLength=512,
            caseSensitive=True
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
        BertEmbeddings
            The restored model
        """
        from sparknlp.internal import _SpanBertCorefLoader
        jModel = _SpanBertCorefLoader(folder, spark_session._jsparkSession)._java_obj
        return SpanBertCorefModel(java_model=jModel)

    @staticmethod
    def pretrained(name="spanbert_base_coref_model", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "small_bert_L2_768"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        BertEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SpanBertCorefModel, name, lang, remote_loc)

