#  Copyright 2017-2025 John Snow Labs
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
import sparknlp
from sparknlp.internal import ExtendedJavaWrapper

class Partition(ExtendedJavaWrapper):

    def  __init__(self, **kwargs):
        self.spark = sparknlp.start()
        params = {}
        for key, value in kwargs.items():
            try:
                params[key] = str(value)
            except Exception as e:
                raise ValueError(f"Invalid value for key '{key}': Cannot cast {type(value)} to string. Original error: {e}")

        super(Partition, self).__init__("com.johnsnowlabs.partition.Partition", params)

    def partition(self, path, headers=None):
        if headers is None:
            headers = {}
        jdf = self._java_obj.partition(path, headers)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    def partition_urls(self, path, headers=None):
        if headers is None:
            headers = {}
        jdf = self._java_obj.partitionUrlsJava(path, headers)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    def partition_text(self, text):
        jdf = self._java_obj.partitionText(text)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe