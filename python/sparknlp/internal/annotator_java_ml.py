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
"""Contains utility classes for annotators."""

from pyspark.ml.util import JavaMLReader, JavaMLReadable


class AnnotatorJavaMLReadable(JavaMLReadable):
    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return AnnotatorJavaMLReader(cls())


class AnnotatorJavaMLReader(JavaMLReader):
    @classmethod
    def _java_loader_class(cls, clazz):
        if hasattr(clazz, '_java_class_name') and clazz._java_class_name is not None:
            return clazz._java_class_name
        else:
            return JavaMLReader._java_loader_class(clazz)
