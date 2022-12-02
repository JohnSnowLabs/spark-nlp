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
"""Contains classes for extensions of the PySpark JavaWrapper."""

from pyspark import SparkContext
from pyspark.ml.wrapper import JavaWrapper
from pyspark.sql import DataFrame
from distutils.version import LooseVersion


class ExtendedJavaWrapper(JavaWrapper):

    def __init__(self, java_obj, *args):
        super(ExtendedJavaWrapper, self).__init__(java_obj)
        self.sc = SparkContext._active_spark_context
        self._java_obj = self.new_java_obj(java_obj, *args)
        self.java_obj = self._java_obj

    def __del__(self):
        pass

    def apply(self):
        return self._java_obj

    def new_java_obj(self, java_class, *args):
        return self._new_java_obj(java_class, *args)

    def new_java_array(self, pylist, java_class):
        """
        ToDo: Inspired from spark 2.0. Review if spark changes
        """
        java_array = self.sc._gateway.new_array(java_class, len(pylist))
        for i in range(len(pylist)):
            java_array[i] = pylist[i]
        return java_array

    def new_java_array_string(self, pylist):
        java_array = self._new_java_array(pylist, self.sc._gateway.jvm.java.lang.String)
        return java_array

    def new_java_array_integer(self, pylist):
        java_array = self._new_java_array(pylist, self.sc._gateway.jvm.java.lang.Integer)
        return java_array

    def spark_version(self):
        return self.sc.version

    def getDataFrame(self, spark, jdf):
        if LooseVersion(self.spark_version()) >= LooseVersion("3.3.0"):
            return DataFrame(jdf, spark._getActiveSessionOrCreate())
        else:
            return DataFrame(jdf, spark._wrapped)