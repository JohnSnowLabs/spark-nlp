#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql import DataFrame
from sparknlp.annotation import Annotation


def map_annotations(f, output_type: DataType):
    return udf(
        lambda content: [ Annotation.toRow(a) for a in f([Annotation.fromRow(r) for r in content])],
        output_type
    )


def map_annotations_strict(f):
    return udf(
        lambda content: [ Annotation.toRow(a) for a in f([Annotation.fromRow(r) for r in content])],
        ArrayType(Annotation.dataType())
    )


def map_annotations_col(dataframe: DataFrame, f, column, output_column, output_type):
    return dataframe.withColumn(output_column, map_annotations(f, output_type)(column))


def filter_by_annotations_col(dataframe, f, column):
    this_udf = udf(
        lambda content: f(content),
        BooleanType()
    )
    return dataframe.filter(this_udf(column))


def explode_annotations_col(dataframe: DataFrame, column, output_column):
    from pyspark.sql.functions import explode
    return dataframe.withColumn(output_column, explode(column))
