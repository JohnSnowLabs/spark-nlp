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
from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param

from sparknlp.common import AnnotatorType
from sparknlp.internal import AnnotatorTransformer

class Reader2Doc(AnnotatorTransformer):
    name = 'Reader2Doc'
    outputAnnotatorType = AnnotatorType.DOCUMENT

    contentPath = Param(
        Params._dummy(),
        "contentPath",
        "contentPath path to files to read",
        typeConverter=TypeConverters.toString
    )

    outputCol = Param(
        Params._dummy(),
        "outputCol",
        "output column name",
        typeConverter=TypeConverters.toString
    )

    contentType = Param(
        Params._dummy(),
        "contentType",
        "Set the content type to load following MIME specification",
        typeConverter=TypeConverters.toString
    )

    explodeDocs = Param(
        Params._dummy(),
        "explodeDocs",
        "whether to explode the documents into separate rows",
        typeConverter=TypeConverters.toBoolean
    )

    @keyword_only
    def __init__(self):
        super(Reader2Doc, self).__init__(classname="com.johnsnowlabs.reader.Reader2Doc")
        self._setDefault(outputCol="document")

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setContentPath(self, value):
        """Sets content path.

        Parameters
        ----------
        value : str
            contentPath path to files to read
        """
        return self._set(contentPath=value)

    def setContentType(self, value):
        return self._set(contentType=value)

    def setExplodeDocs(self, value):
        return self._set(explodeDocs=value)

    def setOutputCol(self, value):
        """Sets output column name.

        Parameters
        ----------
        value : str
            Name of the Output Column
        """
        return self._set(outputCol=value)