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
"""Contains the base classes for Annotators based on the Spark Transformer."""

from pyspark import keyword_only
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer

from sparknlp.internal.params_getters_setters import ParamsGettersSetters
from sparknlp.internal.annotator_java_ml import AnnotatorJavaMLReadable


class AnnotatorTransformer(JavaTransformer, AnnotatorJavaMLReadable, JavaMLWritable, ParamsGettersSetters):

    outputAnnotatorType = None

    @keyword_only
    def __init__(self, classname):
        super(AnnotatorTransformer, self).__init__()
        kwargs = self._input_kwargs
        if 'classname' in kwargs:
            kwargs.pop('classname')
        self.setParams(**kwargs)
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)

