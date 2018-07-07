from sparknlp.internal import _OcrCreateDataset, _OcrCreateMap
from pyspark.sql import SparkSession, DataFrame


class OcrHelper:
    @staticmethod
    def createDataset(spark, input_path, output_col, metadata_col):
        if type(spark) != SparkSession:
            raise Exception("spark must be SparkSession")
        return DataFrame(_OcrCreateDataset(spark._jsparkSession, input_path, output_col, metadata_col).apply(), spark)

    @staticmethod
    def createMap(input_path):
        return _OcrCreateMap(input_path).apply()
