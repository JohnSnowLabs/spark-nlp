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
"""Contains the base classes for Annotator Models."""

from pyspark import keyword_only
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaModel

import sparknlp.internal as _internal
from sparknlp.common import AnnotatorProperties


class AnnotatorModel(JavaModel, _internal.AnnotatorJavaMLReadable, JavaMLWritable, AnnotatorProperties,
                     _internal.ParamsGettersSetters):

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @keyword_only
    def __init__(self, classname, java_model=None):
        super(AnnotatorModel, self).__init__(java_model=java_model)
        if classname and not java_model:
            self.__class__._java_class_name = classname
            self._java_obj = self._new_java_obj(classname, self.uid)
        if java_model is not None:
            self._transfer_params_from_java()
        self._setDefault(lazyAnnotator=False)

    def __init_subclass__(cls, **kwargs):
        for required in ('inputAnnotatorTypes', 'outputAnnotatorType'):
            if not getattr(cls, required):
                raise TypeError(f"Can't instantiate class {cls.__name__} without {required} attribute defined")

        return super().__init_subclass__(**kwargs)
