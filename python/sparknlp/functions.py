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
    output_type : :class:`pyspark.sql.types.DataType`
        Output type of the data

    Returns
    -------
    :func:`pyspark.sql.functions.udf`
        Spark UserDefinedFunction (udf)

    Examples
    --------
    >>> from sparknlp.pretrained import PretrainedPipeline
    >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
    >>> data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
    >>> result = explain_document_pipeline.transform(data)

    The array type must be provided in order to tell Spark the expected output
    type of our column. We are using an Annotation array here.

    >>> from sparknlp.functions import *
    >>> def nnp_tokens(annotations: List[Row]):
    ...     return list(
    ...         filter(lambda annotation: annotation.result == 'NNP', annotations)
    ...     )
    >>> result.select(
    ...     map_annotations(nnp_tokens, Annotation.arrayType())('pos').alias("nnp")
    ... ).selectExpr("explode(nnp) as nnp").show(truncate=False)
    +-----------------------------------------+
    |nnp                                      |
    +-----------------------------------------+
    |[pos, 0, 2, NNP, [word -> U.N], []]      |
    |[pos, 14, 18, NNP, [word -> Epeus], []]  |
    |[pos, 30, 36, NNP, [word -> Baghdad], []]|
    +-----------------------------------------+
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
    output_type : :class:`pyspark.sql.types.DataType`
        Output type of the data

    Returns
    -------
    :func:`pyspark.sql.functions.udf`
        Spark UserDefinedFunction (udf)
    """
    return udf(
        lambda cols: [Annotation.toRow(item) for item in f([Annotation.fromRow(r) for col in cols for r in col])],
        output_type
    )

def map_annotations_strict(f):
    """Creates a Spark UDF to map over an Annotator's results, for which the
    return type is explicitly defined as a `Annotation.dataType()`.

    Parameters
    ----------
    f : function
        The function to be applied over the results

    Returns
    -------
    :func:`pyspark.sql.functions.udf`
        Spark UserDefinedFunction (udf)

    Examples
    --------
    >>> from sparknlp.pretrained import PretrainedPipeline
    >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
    >>> data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
    >>> result = explain_document_pipeline.transform(data)
    >>> def nnp_tokens(annotations):
    ...     return list(
    ...         filter(lambda annotation: annotation.result == 'NNP', annotations)
    ...     )
    >>> result.select(
    ...     map_annotations_strict(nnp_tokens)('pos').alias("nnp")
    ... ).selectExpr("explode(nnp) as nnp").show(truncate=False)
    +-----------------------------------------+
    |nnp                                      |
    +-----------------------------------------+
    |[pos, 0, 2, NNP, [word -> U.N], []]      |
    |[pos, 14, 18, NNP, [word -> Epeus], []]  |
    |[pos, 30, 36, NNP, [word -> Baghdad], []]|
    +-----------------------------------------+
    """
    return udf(
        lambda content: [ Annotation.toRow(a) for a in f([Annotation.fromRow(r) for r in content])],
        ArrayType(Annotation.dataType())
    )


def map_annotations_col(dataframe: DataFrame, f, column: str, output_column: str, annotatyon_type: str,
                        output_type: DataType = Annotation.arrayType()):
    """Creates a Spark UDF to map over a column of Annotation results.

    Parameters
    ----------
    dataframe : DataFrame
        Input DataFrame
    f : function
        Function to apply to the column
    column : str
        Name of the input column
    output_column : str
        Name of the output column
    annotatyon_type : str
        Annotator type
    output_type : DataType, optional
        Output type, by default Annotation.arrayType()

    Returns
    -------
    :class:`pyspark.sql.DataFrame`
        Transformed DataFrame

    Examples
    --------
    >>> from sparknlp.pretrained import PretrainedPipeline
    >>> from sparknlp.functions import *
    >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
    >>> data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
    >>> result = explain_document_pipeline.transform(data)
    >>> chunks_df = map_annotations_col(
    ...     result,
    ...     lambda x: [
    ...         Annotation("chunk", a.begin, a.end, a.result, a.metadata, a.embeddings)
    ...         for a in x
    ...     ],
    ...     "pos",
    ...     "pos_chunk",
    ...     "chunk",
    ... )
    >>> chunks_df.selectExpr("explode(pos_chunk)").show()
    +--------------------+
    |                 col|
    +--------------------+
    |[chunk, 0, 2, NNP...|
    |[chunk, 3, 3, ., ...|
    |[chunk, 5, 12, JJ...|
    |[chunk, 14, 18, N...|
    |[chunk, 20, 24, V...|
    |[chunk, 26, 28, I...|
    |[chunk, 30, 36, N...|
    |[chunk, 37, 37, ....|
    +--------------------+
    """
    return dataframe.withColumn(output_column, map_annotations(f, output_type)(column).alias(output_column, metadata={
        'annotatorType': annotatyon_type}))

def map_annotations_cols(dataframe: DataFrame, f, columns: list, output_column: str, annotatyon_type: str,
                        output_type: DataType = Annotation.arrayType()):
    """Creates a Spark UDF to map over multiple columns of Annotation results.

    Parameters
    ----------
    dataframe : DataFrame
        Input DataFrame
    f : function
        Function to apply to the column
    columns : list
        Name of the input column
    output_column : str
        Name of the output column
    annotatyon_type : str
        Annotator type
    output_type : DataType, optional
        Output type, by default Annotation.arrayType()

    Returns
    -------
    :class:`pyspark.sql.DataFrame`
        Transformed DataFrame

    Examples
    --------
    >>> from sparknlp.pretrained import PretrainedPipeline
    >>> from sparknlp.functions import *
    >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
    >>> data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
    >>> result = explain_document_pipeline.transform(data)
    >>> chunks_df = map_annotations_cols(
    ...     result,
    ...     lambda x: [
    ...         Annotation("tag", a.begin, a.end, a.result, a.metadata, a.embeddings)
    ...         for a in x
    ...     ],
    ...     ["pos", "ner"],
    ...     "tags",
    ...     "chunk"
    ... )
    >>> chunks_df.selectExpr("explode(tags)").show(truncate=False)
    +-------------------------------------------+
    |col                                        |
    +-------------------------------------------+
    |[tag, 0, 2, NNP, [word -> U.N], []]        |
    |[tag, 3, 3, ., [word -> .], []]            |
    |[tag, 5, 12, JJ, [word -> official], []]   |
    |[tag, 14, 18, NNP, [word -> Epeus], []]    |
    |[tag, 20, 24, VBZ, [word -> heads], []]    |
    |[tag, 26, 28, IN, [word -> for], []]       |
    |[tag, 30, 36, NNP, [word -> Baghdad], []]  |
    |[tag, 37, 37, ., [word -> .], []]          |
    |[tag, 0, 2, B-ORG, [word -> U.N], []]      |
    |[tag, 3, 3, O, [word -> .], []]            |
    |[tag, 5, 12, O, [word -> official], []]    |
    |[tag, 14, 18, B-PER, [word -> Ekeus], []]  |
    |[tag, 20, 24, O, [word -> heads], []]      |
    |[tag, 26, 28, O, [word -> for], []]        |
    |[tag, 30, 36, B-LOC, [word -> Baghdad], []]|
    |[tag, 37, 37, O, [word -> .], []]          |
    +-------------------------------------------+
    """
    return dataframe.withColumn(output_column, map_annotations_array(f, output_type)(array(*columns)).alias(output_column, metadata={
        'annotatorType': annotatyon_type}))


def filter_by_annotations_col(dataframe, f, column):
    """Applies a filter over a column of Annotations.

    Parameters
    ----------
    dataframe : DataFrame
        Input DataFrame
    f : function
        Filter function
    column : str
        Name of the column

    Returns
    -------
    :class:`pyspark.sql.DataFrame`
        Filtered DataFrame

    Examples
    --------
    >>> from sparknlp.pretrained import PretrainedPipeline
    >>> from sparknlp.functions import *
    >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
    >>> data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
    >>> result = explain_document_pipeline.transform(data)
    >>> def filter_pos(annotation: Annotation):
    ...     return annotation.result == "NNP"
    >>> filter_by_annotations_col(
    ...     explode_annotations_col(result, "pos", "pos"), filter_pos, "pos"
    ... ).select("pos").show(truncate=False)
    +-----------------------------------------+
    |pos                                      |
    +-----------------------------------------+
    |[pos, 0, 2, NNP, [word -> U.N], []]      |
    |[pos, 14, 18, NNP, [word -> Epeus], []]  |
    |[pos, 30, 36, NNP, [word -> Baghdad], []]|
    +-----------------------------------------+
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
    :class:`pyspark.sql.DataFrame`
        Transformed DataFrame

    Examples
    --------
    >>> from sparknlp.pretrained import PretrainedPipeline
    >>> from sparknlp.functions import *
    >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
    >>> data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
    >>> result = explain_document_pipeline.transform(data)
    >>> result.select("pos.result").show(truncate=False)
    +----------------------------------+
    |result                            |
    +----------------------------------+
    |[NNP, ., JJ, NNP, VBZ, IN, NNP, .]|
    +----------------------------------+
    >>> explode_annotations_col(result, "pos", "pos").select("pos.result").show()
    +------+
    |result|
    +------+
    |   NNP|
    |     .|
    |    JJ|
    |   NNP|
    |   VBZ|
    |    IN|
    |   NNP|
    |     .|
    +------+
    """
    from pyspark.sql.functions import explode
    return dataframe.withColumn(output_column, explode(column))
