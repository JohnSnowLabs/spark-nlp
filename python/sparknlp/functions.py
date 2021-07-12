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

"""Contains helper functions to assist in transforming Annotation results.
"""

from pyspark.sql.functions import udf, array
from pyspark.sql.types import *
from pyspark.sql import DataFrame
from sparknlp.annotation import Annotation


def map_annotations(f, output_type: DataType):
    """Creates a Spark UDF to map over an Annotator's results.

    Parameters
    ----------
    f : function
        The function to be applied over the results
    output_type : DataType
        Output type of the data

    Returns
    -------
    :class:`UserDefinedFunction`
        Spark UserDefinedFunction (udf)
    """
    return udf(
        lambda content: [ Annotation.toRow(a) for a in f([Annotation.fromRow(r) for r in content])],
        output_type
    )

def map_annotations_array(f, output_type: DataType):
    """Creates a Spark UDF to map over an Annotator's array results.

    Parameters
    ----------
    f : function
        The function to be applied over the results
    output_type : DataType
        Output type of the data

    Returns
    -------
    :class:`UserDefinedFunction`
        Spark UserDefinedFunction (udf)
    """
    return udf(
        lambda cols: [Annotation.toRow(item) for item in f([Annotation.fromRow(r) for col in cols for r in col])],
        output_type
    )

def map_annotations_strict(f):
    """Creates a Spark UDF to map over an Annotator's results, for which the return type is explicitly defined as
    a :func:`Annotation.dataType`.

    Parameters
    ----------
    f : function
        The function to be applied over the results

    Returns
    -------
    :class:`UserDefinedFunction`
        Spark UserDefinedFunction (udf)
    """
    return udf(
        lambda content: [ Annotation.toRow(a) for a in f([Annotation.fromRow(r) for r in content])],
        ArrayType(Annotation.dataType())
    )


def map_annotations_col(dataframe: DataFrame, f, column: str, output_column: str, annotatyon_type: str,
                        output_type: DataType = Annotation.arrayType()):
    """Creates a Spark UDF to map over an Annotator's array results, for which the return type is explicitly defined as
    a :class:`Annotation`.

    Parameters
    ----------
    f : function
        The function to be applied over the results
    output_type : DataType
        Output type of the data

    Returns
    -------
    :class:`UserDefinedFunction`
        Spark UserDefinedFunction (udf)
    """
    return dataframe.withColumn(output_column, map_annotations(f, output_type)(column).alias(output_column, metadata={
        'annotatorType': annotatyon_type}))

def map_annotations_cols(dataframe: DataFrame, f, columns: list, output_column: str, annotatyon_type: str,
                        output_type: DataType = Annotation.arrayType()):
    """Maps a function over a column of :class:`sparknlp.annotation.Annotation` to create a new column.

    Parameters
    ----------
    dataframe : DataFrame
        The Spark Dataframe containing output Annotations
    f : function
        The function to be applied over the results
    columns : list
        Columns to apply the filter over
    output_column : str
        Name of the output Column
    annotatyon_type : str
        Annotator Type
    output_type : DataType, optional
        Output Annotation schema, by default Annotation.arrayType()

    Returns
    -------
    :class:`DataFrame`
        Spark DataFrame with new column.
    """

    return dataframe.withColumn(output_column, map_annotations_array(f, output_type)(array(*columns)).alias(output_column, metadata={
        'annotatorType': annotatyon_type}))


def filter_by_annotations_col(dataframe, f, column):
    """Applies a filter over a column of Annotations.

    Parameters
    ----------
    f : function
        The function to be applied over the results.
    column : str
        Name of the column.

    Returns
    -------
    :class:`DataFrame`
        The filtered Dataframe
    """
    this_udf = udf(
        lambda content: f(content),
        BooleanType()
    )
    return dataframe.filter(this_udf(column))


def explode_annotations_col(dataframe: DataFrame, column, output_column):
    """Explodes an Annotation column, putting each result onto a separate row.

    Parameters
    ----------
    dataframe : DataFrame
        The Spark DataFrame containing output Annotations
    column : str
        Name of the column
    output_column : str
        Name of the output column

    Returns
    -------
    :class:`DataFrame`
        The filtered Dataframe
    """
    from pyspark.sql.functions import explode
    return dataframe.withColumn(output_column, explode(column))
