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

from sparknlp.common import *

class BLIPForQuestionAnswering(AnnotatorModel,
                               HasBatchedAnnotateImage,
                               HasImageFeatureProperties,
                               HasEngine,
                               HasCandidateLabelsProperties,
                               HasRescaleFactor):

    name = "BLIPForQuestionAnswering"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with "
                             "config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    maxSentenceLength = Param(Params._dummy(),
                            "maxSentenceLength",
                            "Maximum sentence length that the annotator will process. Above this, the sentence is skipped",
                            typeConverter=TypeConverters.toInt)

    def setMaxSentenceSize(self, value):
        """Sets Maximum sentence length that the annotator will process, by
        default 50.

        Parameters
        ----------
        value : int
            Maximum sentence length that the annotator will process
        """
        return self._set(maxSentenceLength=value)


    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.BLIPForQuestionAnswering",
                 java_model=None):
        super(BLIPForQuestionAnswering, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=2,
            size=224,
            maxSentenceLength=50
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
        CLIPForZeroShotClassification
            The restored model
        """
        from sparknlp.internal import _BLIPForQuestionAnswering
        jModel = _BLIPForQuestionAnswering(folder, spark_session._jsparkSession)._java_obj
        return BLIPForQuestionAnswering(java_model=jModel)

    @staticmethod
    def pretrained(name="blip_vqa_base", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "blip_vqa_tf"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        CLIPForZeroShotClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BLIPForQuestionAnswering, name, lang, remote_loc)