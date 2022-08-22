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

"""Contains classes concerning ViTForImageClassification."""

from sparknlp.common import *


class ViTForImageClassification(AnnotatorModel,
                                HasBatchedAnnotateImage,
                                HasImageFeatureProperties):
    
    name = "ViTForImageClassification"

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with "
                             "config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    def getClasses(self):
        """
        Returns labels used to train this model
        """
        return self._call_java("getClasses")

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.ViTForImageClassification",
                 java_model=None):
        super(ViTForImageClassification, self).__init__(
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
        ViTForImageClassification
            The restored model
        """
        from sparknlp.internal import _ViTForImageClassification
        jModel = _ViTForImageClassification(folder, spark_session._jsparkSession)._java_obj
        return ViTForImageClassification(java_model=jModel)

    @staticmethod
    def pretrained(name="image_classifier_vit_base_patch16_224", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "image_classifier_vit_base_patch16_224"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        ViTForImageClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ViTForImageClassification, name, lang, remote_loc)
