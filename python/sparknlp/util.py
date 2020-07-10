import sparknlp.internal as _internal


def get_config_path():
    return _internal._ConfigLoaderGetter().apply()


class CoNLLGenerator:
    @staticmethod
    def exportConllFiles(spark, files_path, pipeline, output_path):
        _internal._CoNLLGeneratorExport(spark, files_path, pipeline, output_path).apply()

    @staticmethod
    def exportConllFiles(dataframe, output_path):
        _internal._CoNLLGeneratorExport(dataframe, output_path).apply()
