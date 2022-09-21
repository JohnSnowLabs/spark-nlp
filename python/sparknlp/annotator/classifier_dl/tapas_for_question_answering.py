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
from sparknlp.annotator.classifier_dl import BertForQuestionAnswering


class TapasForQuestionAnswering(BertForQuestionAnswering):
    name = "TapasForQuestionAnswering"

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.TapasForQuestionAnswering",
                 java_model=None):
        super(TapasForQuestionAnswering, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=False
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
        TapasForQuestionAnswering
            The restored model
        """
        from sparknlp.internal import _TapasForQuestionAnsweringLoader
        jModel = _TapasForQuestionAnsweringLoader(folder, spark_session._jsparkSession)._java_obj
        return TapasForQuestionAnswering(java_model=jModel)

    @staticmethod
    def pretrained(name="", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "TODO"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        TapasForQuestionAnswering
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TapasForQuestionAnswering, name, lang, remote_loc)
