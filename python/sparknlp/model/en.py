from sparknlp.downloader import ResourceDownloader
from sparknlp.base import LightPipeline
import pyspark


class _Model:
    @staticmethod
    def _annotate(model_class, name, target, target_column=None):
        if not model_class.model:
            model_class.model = ResourceDownloader().downloadPipeline(name, "en")
        if type(target) is pyspark.sql.dataframe.DataFrame:
            if not target_column:
                raise Exception("target_column argument needed when targeting a DataFrame")
            return model_class.model.transform(target.withColumnRenamed(target_column, "text"))
        elif type(target) is list or type(target) is str:
            pip = LightPipeline(model_class.model)
            return pip.annotate(target)


class Basic:
    model = None

    @staticmethod
    def annotate(target, target_column=None):
        return _Model._annotate(Basic, "pipeline_basic", target, target_column)


class Advanced:
    model = None

    @staticmethod
    def annotate(target, target_column=None):
        return _Model._annotate(Basic, "advanced_basic", target, target_column)
