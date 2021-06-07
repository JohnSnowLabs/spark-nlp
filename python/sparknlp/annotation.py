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

from pyspark.sql.types import *


class Annotation:

    def __init__(self, annotator_type, begin, end, result, metadata, embeddings):
        self.annotator_type = annotator_type
        self.begin = begin
        self.end = end
        self.result = result
        self.metadata = metadata
        self.embeddings = embeddings

    def copy(self, result):
        return Annotation(self.annotator_type, self.begin, self.end, result, self.metadata, self.embeddings)

    def __str__(self):
        return "Annotation(%s, %i, %i, %s, %s)" % (
            self.annotator_type,
            self.begin,
            self.end,
            self.result,
            str(self.metadata)
        )

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def dataType():
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
        return ArrayType(Annotation.dataType())

    @staticmethod
    def fromRow(row):
        return Annotation(row.annotatorType, row.begin, row.end, row.result, row.metadata, row.embeddings)

    @staticmethod
    def toRow(annotation):
        from pyspark.sql import Row
        return Row(annotation.annotator_type, annotation.begin, annotation.end, annotation.result, annotation.metadata, annotation.embeddings)


