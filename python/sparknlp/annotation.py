from pyspark.sql.types import *


class Annotation:

    def __init__(self, annotator_type, begin, end, result, metadata, embeddings):
        self.annotator_type = annotator_type
        self.begin = begin
        self.end = end
        self.result = result
        self.metadata = metadata
        self.embeddings = embeddings

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
            StructField('annotator_type', StringType(), False),
            StructField('begin', IntegerType(), False),
            StructField('end', IntegerType(), False),
            StructField('result', StringType(), False),
            StructField('metadata', MapType(StringType(), StringType()), False),
            StructField('embeddings', ArrayType(FloatType()), False)
        ])

    @staticmethod
    def arrayType():
        return ArrayType(Annotation.dataType())
