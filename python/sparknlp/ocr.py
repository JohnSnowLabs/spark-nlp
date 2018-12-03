import sparknlp.internal as _int
from pyspark.sql import SparkSession, DataFrame


class OcrHelper:
    @staticmethod
    def createDataset(spark, input_path, output_col, metadata_col):
        if type(spark) != SparkSession:
            raise Exception("spark must be SparkSession")
        return DataFrame(_int._OcrCreateDataset(spark._jsparkSession, input_path, output_col, metadata_col).apply(), spark)

    @staticmethod
    def createMap(input_path):
        return _int._OcrCreateMap(input_path).apply()
    
    @staticmethod
    def setMinTextLayer(value):
        return _int._OcrSetMinTextLayer(value).apply()

    @staticmethod
    def getMinTextLayer():
        return _int._OcrGetMinTextLayer().apply()
    
    @staticmethod
    def setEngineMode(mode):
        return _int._OcrSetEngineMode(mode).apply()
    
    @staticmethod
    def getEngineMode():
        return _int._OcrGetEngineMode().apply()

    @staticmethod
    def setPageIteratorLevel(level):
        return _int._OcrSetPageIteratorLevel(level).apply()

    @staticmethod
    def getPageIteratorLevel():
        return _int._OcrGetPageIteratorLevel().apply()

    @staticmethod
    def setScalingFactor(factor):
        return _int._OcrSetScalingFactor(factor).apply()

    @staticmethod
    def setSplitPages(value):
        return _int._OcrSetSplitPages(value).apply()

    @staticmethod
    def getSplitPages():
        return _int._OcrGetSplitPages().apply()

    @staticmethod
    def useErosion(use, k_size=2, k_shape=0):
        return _int._OcrUseErosion(use, k_size, k_shape).apply()
