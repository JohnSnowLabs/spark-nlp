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
"""Contains various utilities."""


import sparknlp.internal as _internal
import numpy as np
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BinaryType


def get_config_path():
    return _internal._ConfigLoaderGetter().apply()


class CoNLLGenerator:
    @staticmethod
    def exportConllFiles(*args):
        num_args = len(args)
        if num_args == 2:
            _internal._CoNLLGeneratorExportFromDataFrame(*args).apply()
        elif num_args == 3:
            _internal._CoNLLGeneratorExportFromDataFrameAndField(*args).apply()
        elif num_args == 4:
            _internal._CoNLLGeneratorExportFromTargetAndPipeline(*args).apply()
        else:
            raise NotImplementedError(f"No exportConllFiles alternative takes {num_args} parameters")


class EmbeddingsDataFrameUtils:
    """
    Utility for creating DataFrames compatible with multimodal embedding models (e.g., E5VEmbeddings) for text-only scenarios.
    Provides:
      - imageSchema: the expected schema for Spark image DataFrames
      - emptyImageRow: a dummy image row for text-only embedding
    """
    imageSchema = StructType([
        StructField(
            "image",
            StructType([
                StructField("origin", StringType(), True),
                StructField("height", IntegerType(), True),
                StructField("width", IntegerType(), True),
                StructField("nChannels", IntegerType(), True),
                StructField("mode", IntegerType(), True),
                StructField("data", BinaryType(), True),
            ]),
        )
    ])
    emptyImageRow = Row(Row("", 0, 0, 0, 0, bytes()))
