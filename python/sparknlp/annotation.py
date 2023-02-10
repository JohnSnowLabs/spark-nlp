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

"""Contains the Annotation data format
"""

from pyspark.sql.types import *


class Annotation:
    """Represents the output of Spark NLP Annotators and their details.

    Parameters
    ----------
    annotator_type : str
        The type of the output of the annotator. Possible values are ``DOCUMENT,
        TOKEN, WORDPIECE, WORD_EMBEDDINGS, SENTENCE_EMBEDDINGS, CATEGORY, DATE,
        ENTITY, SENTIMENT, POS, CHUNK, NAMED_ENTITY, NEGEX, DEPENDENCY,
        LABELED_DEPENDENCY, LANGUAGE, KEYWORD, DUMMY``.
    begin : int
        The index of the first character under this annotation.
    end : int
        The index of the last character under this annotation.
    result : str
        The resulting string of the annotation.
    metadata : dict
        Associated metadata for this annotation
    embeddings : list
        Embeddings vector where applicable
    """

    def __init__(self, annotatorType, begin, end, result, metadata, embeddings):
        self.annotatorType = annotatorType
        self.begin = begin
        self.end = end
        self.result = result
        self.metadata = metadata
        self.embeddings = embeddings

    def copy(self, result):
        """Creates new Annotation with a different result, containing all
        settings of this Annotation.

        Parameters
        ----------
        result : str
            The result of the annotation that should be copied.

        Returns
        -------
        Annotation
            Newly created Annotation
        """
        return Annotation(self.annotatorType, self.begin, self.end, result, self.metadata, self.embeddings)

    def __str__(self):
        return "Annotation(%s, %i, %i, %s, %s, %s)" % (
            self.annotatorType,
            self.begin,
            self.end,
            self.result,
            str(self.metadata),
            str(self.embeddings)
        )

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        same_annotator_type = self.annotatorType == other.annotatorType
        same_result = self.result == other.result
        same_begin = self.begin == other.begin
        same_end = self.end == other.end
        same_metadata = dict(self.metadata) == other.metadata
        same_embeddings = self.embeddings == other.embeddings

        same_annotation = \
            same_annotator_type and same_result and same_begin and same_end and same_metadata and same_embeddings

        return same_annotation

    @staticmethod
    def dataType():
        """Returns a Spark `StructType`, that represents the schema of the
        Annotation.

        The Schema looks like::

            struct (containsNull = True)
            |-- annotatorType: string (nullable = False)
            |-- begin: integer (nullable = False)
            |-- end: integer (nullable = False)
            |-- result: string (nullable = False)
            |-- metadata: map (nullable = False)
            |    |-- key: string
            |    |-- value: string (valueContainsNull = True)
            |-- embeddings: array (nullable = False)
            |    |-- element: float (containsNull = False)

        Returns
        -------
        :class:`pyspark.sql.types.StructType`
            Spark Schema of the Annotation
        """
        return StructType([
            StructField('annotatorType', StringType(), False),
            StructField('begin', IntegerType(), False),
            StructField('end', IntegerType(), False),
            StructField('result', StringType(), False),
            StructField('metadata', MapType(StringType(), StringType()), False),
            StructField('embeddings', ArrayType(FloatType()), False)
        ])

    @staticmethod
    def arrayType():
        """Returns a Spark `ArrayType`, that contains the `dataType` of the
        annotation.

        Returns
        -------
        :class:`pyspark.sql.types.ArrayType`
            ArrayType with the Annotation data type embedded.
        """
        return ArrayType(Annotation.dataType())

    @staticmethod
    def fromRow(row):
        """Creates a Annotation from a Spark `Row`.

        Parameters
        ----------
        row : :class:`pyspark.sql.Row`
            Spark row containing columns for ``annotatorType, begin, end,
            result, metadata, embeddings``.

        Returns
        -------
        Annotation
            The new Annotation.
        """
        return Annotation(row.annotatorType, row.begin, row.end, row.result, row.metadata, row.embeddings)

    @staticmethod
    def toRow(annotation):
        """Transforms an Annotation to a Spark `Row`.

        Parameters
        ----------
        annotation : Annotation
            The Annotation to be transformed.

        Returns
        -------
        :class:`pyspark.sql.Row`
            The new Row.
        """
        from pyspark.sql import Row
        return Row(annotation.annotatorType, annotation.begin, annotation.end, annotation.result, annotation.metadata,
                   annotation.embeddings)
