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
"""Contains classes for the AudioAssembler."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param

from sparknlp.common import AnnotatorType
from sparknlp.internal import AnnotatorTransformer


class AudioAssembler(AnnotatorTransformer):
    """Prepares Floats or Doubles from a processed audio file(s)
    This component is needed to process audio.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``NONE``               ``AUDIO``
    ====================== ======================

    Parameters
    ----------
    inputCol
        Input column name
    outputCol
        Output column name

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from pyspark.ml import Pipeline
    >>> data = spark.read.option("inferSchema", value = True)\
                    .parquet("./tmp/librispeech_asr_dummy_clean_audio_array_parquet")\
                    .select($"float_array".cast("array<float>").as("audio_content"))
    >>> audioAssembler = AudioAssembler().setInputCol("audio_content").setOutputCol("audio_assembler")
    >>> result = audioAssembler.transform(data)
    >>> result.select("audio_assembler").show()
    >>> result.select("audio_assembler").printSchema()
    root
     |-- audio_content: array (nullable = true)
     |    |-- element: float (containsNull = true)
    """

    inputCol = Param(Params._dummy(), "inputCol", "input column name", typeConverter=TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "output column name", typeConverter=TypeConverters.toString)
    name = 'AudioAssembler'

    outputAnnotatorType = AnnotatorType.AUDIO

    @keyword_only
    def __init__(self):
        super(AudioAssembler, self).__init__(classname="com.johnsnowlabs.nlp.AudioAssembler")
        self._setDefault(outputCol="audio_assembler", inputCol='audio')

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        """Sets input column name.

        Parameters
        ----------
        value : str
            Name of the input column that has audio in format of Array[Float] or Array[Double]
        """
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        """Sets output column name.

        Parameters
        ----------
        value : str
            Name of the Output Column
        """
        return self._set(outputCol=value)

    def getOutputCol(self):
        """Gets output column name of annotations."""
        return self.getOrDefault(self.outputCol)
