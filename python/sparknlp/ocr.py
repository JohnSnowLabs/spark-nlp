from sparknlp.internal import _OcrCreateDataset, _OcrCreateMap


class OcrHelper:
    @staticmethod
    def createDataset(spark, input_path):
        return _OcrCreateDataset(spark, input_path).apply()

    @staticmethod
    def createMap(input_path):
        return _OcrCreateMap(input_path).apply()
