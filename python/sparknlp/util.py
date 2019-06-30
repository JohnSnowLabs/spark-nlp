import sparknlp.internal as _internal

from pyspark.ml.util import JavaMLReadable, JavaMLReader


def get_config_path():
    return _internal._ConfigLoaderGetter().apply()




class AnnotatorJavaMLReadable(JavaMLReadable):
    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return AnnotatorJavaMLReader(cls())


class AnnotatorJavaMLReader(JavaMLReader):
    @classmethod
    def _java_loader_class(cls, clazz):
        if hasattr(clazz, '_java_class_name') and clazz._java_class_name is not None:
            return clazz._java_class_name
        else:
            return JavaMLReader._java_loader_class(clazz)


class CoNLLGenerator:
    @staticmethod
    def exportConllFiles(spark, files_path, pipeline, output_path):
        _internal._CoNLLGeneratorExport(spark, files_path, pipeline, output_path).apply()
