#  Copyright 2017-2023 John Snow Labs
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
from pyspark.sql import SparkSession

from sparknlp.internal import ExtendedJavaWrapper


class SpacyToAnnotation(ExtendedJavaWrapper):

    def __init__(self):
        super(SpacyToAnnotation, self).__init__("com.johnsnowlabs.nlp.training.SpacyToAnnotation")

    def readJsonFile(self, spark, jsonFilePath, params=None):
        if params is None:
            params = {}

        jSession = spark._jsparkSession

        jdf = self._java_obj.readJsonFileJava(jSession, jsonFilePath, params)
        annotation_dataset = self.getDataFrame(spark, jdf)
        return annotation_dataset
