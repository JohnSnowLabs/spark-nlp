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
"""Contains the base classes for Annotator Approaches."""

from pyspark import keyword_only
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator

import sparknlp.internal as _internal
from sparknlp.common.annotator_properties import AnnotatorProperties


class AnnotatorApproach(JavaEstimator, JavaMLWritable, _internal.AnnotatorJavaMLReadable, AnnotatorProperties,
                        _internal.ParamsGettersSetters):

    @keyword_only
    def __init__(self, classname):
        _internal.ParamsGettersSetters.__init__(self)
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)
        self._setDefault(lazyAnnotator=False)

    def _create_model(self, java_model):
        raise NotImplementedError('Please implement _create_model in %s' % self)

    def __init_subclass__(cls, **kwargs):
        for required in ('inputAnnotatorTypes', 'outputAnnotatorType'):
            if not getattr(cls, required):
                raise TypeError(f"Can't instantiate class {cls.__name__} without {required} attribute defined")
        return super().__init_subclass__(**kwargs)
