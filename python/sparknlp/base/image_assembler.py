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
"""Contains classes for the ImageAssembler."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import regexp_replace, col

from sparknlp.common import AnnotatorType
from sparknlp.internal import AnnotatorTransformer


class ImageAssembler(AnnotatorTransformer):
    """Prepares images read by Spark into a format that is processable by Spark NLP.
    This component is needed to process images.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``NONE``               ``IMAGE``
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
    >>> data = spark.read.format("image").load("./tmp/images/").toDF("image")
    >>> imageAssembler = ImageAssembler().setInputCol("image").setOutputCol("image_assembler")
    >>> result = imageAssembler.transform(data)
    >>> result.select("image_assembler").show()
    >>> result.select("image_assembler").printSchema()
    root
     |-- image_assembler: array (nullable = true)
     |    |-- element: struct (containsNull = true)
     |    |    |-- annotatorType: string (nullable = true)
     |    |    |-- origin: string (nullable = true)
     |    |    |-- height: integer (nullable = true)
     |    |    |-- width: integer (nullable = true)
     |    |    |-- nChannels: integer (nullable = true)
     |    |    |-- mode: integer (nullable = true)
     |    |    |-- result: binary (nullable = true)
     |    |    |-- metadata: map (nullable = true)
     |    |    |    |-- key: string
     |    |    |    |-- value: string (valueContainsNull = true)
    """

    outputAnnotatorType = AnnotatorType.IMAGE

    inputCol = Param(Params._dummy(), "inputCol", "input column name", typeConverter=TypeConverters.toString)
    textCol = Param(Params._dummy(), "textCol", "text column name", typeConverter=TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "output column name", typeConverter=TypeConverters.toString)
    name = 'ImageAssembler'

    @keyword_only
    def __init__(self):
        super(ImageAssembler, self).__init__(classname="com.johnsnowlabs.nlp.ImageAssembler")
        self._setDefault(outputCol="image_assembler", inputCol='image')

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        """Sets input column name.

        Parameters
        ----------
        value : str
            Name of the input column that has image format loaded via spark.read.format("image").load(PATH)
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

    def setTextCol(self, value):
        """Sets an optional text column name.

        Parameters
        ----------
        value : str
            Name of an optional input text column
        """
        return self._set(inputCol=value)

    @classmethod
    def loadImagesAsBytes(cls, spark: SparkSession, path: str):
        """
        Loads images from a given path and returns them as raw bytes, instead of the default
        OpenCV-compatible format. Supported image types include JPEG, PNG, GIF, and BMP.

        Multimodal inference with llama.cpp requires raw bytes as input.

        Parameters
        ----------
        spark : SparkSession
            The active SparkSession.
        path : str
            The path to the images. Supported image types are JPEG, PNG, GIF, and BMP.

        Returns
        -------
        DataFrame
            A DataFrame containing the images as raw bytes along with their metadata.
        """

        # Replace the path separator in the `origin` field and `path` column, so that they match
        def replace_path(column_name: str):
            return regexp_replace(col(column_name), ":///", ":/")

        # Load the images as metadata with the default Spark image format
        data = (
            spark.read.format("image")
            .option("dropInvalid", True)
            .load(path)
            .withColumn(
                "image", col("image").withField("origin", replace_path("image.origin"))
            )
        )

        # Load the images as raw binary files
        image_bytes = (
            spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.{jpeg,jpg,png,gif,bmp,JPEG,JPG,PNG,GIF,BMP}")
            .option("dropInvalid", True)
            .load(path)
            .withColumn("path", replace_path("path"))
        )

        # Join the two datasets on the file path
        df_joined = data.join(
            image_bytes, data["image.origin"] == image_bytes["path"], "inner"
        )

        # Replace the `data` field of the `image` column with raw bytes
        df_image_replaced = df_joined.withColumn(
            "image", df_joined["image"].withField("data", df_joined["content"])
        )

        return df_image_replaced
