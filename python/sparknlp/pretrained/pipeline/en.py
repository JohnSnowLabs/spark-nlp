from sparknlp.pretrained import ResourceDownloader
from sparknlp.base import LightPipeline
import pyspark


class _Model:
    @staticmethod
    def _annotate_fin(model_class, name, target, target_column=None):
        if not model_class.model_fin:
            model_class.model_fin = ResourceDownloader().downloadPipeline(name, "en")
        return model_class.model_fin.transform(target.withColumnRenamed(target_column, "text"))

    @staticmethod
    def _annotate_ann(model_class, name, target):
        if not model_class.model_ann:
            model_class.model_ann = ResourceDownloader().downloadPipeline(name, "en")
        pip = LightPipeline(model_class.model_ann)
        return pip.annotate(target)


class BasicPipeline:
    model_fin = None
    model_ann = None

    @staticmethod
    def annotate(target, target_column=None):
        if type(target) is pyspark.sql.dataframe.DataFrame:
            if not target_column:
                raise Exception("annotate() target_column arg needed when targeting a DataFrame")
            return _Model._annotate_fin(BasicPipeline, "pipeline_basic_fin", target, target_column)
        elif type(target) is list or type(target) is str:
            return _Model._annotate_ann(BasicPipeline, "pipeline_basic_ann", target)
        else:
            raise Exception("target may be dataframe, string or list of strings")

    @staticmethod
    def retrieve():
        if not BasicPipeline.model_ann:
            BasicPipeline.model_ann = ResourceDownloader().downloadPipeline("pipeline_basic_ann", "en")
        return BasicPipeline.model_ann


class AdvancedPipeline:
    model_fin = None
    model_ann = None

    @staticmethod
    def annotate(target, target_column=None):
        if type(target) is pyspark.sql.dataframe.DataFrame:
            if not target_column:
                raise Exception("annotate() target_column arg needed when targeting a DataFrame")
            return _Model._annotate_fin(AdvancedPipeline, "pipeline_advanced_fin", target, target_column)
        elif type(target) is list or type(target) is str:
            return _Model._annotate_ann(AdvancedPipeline, "pipeline_advanced_ann", target)
        else:
            raise Exception("target may be dataframe, string or list of strings")

    @staticmethod
    def retrieve():
        if not BasicPipeline.model_ann:
            AdvancedPipeline.model_ann = ResourceDownloader().downloadPipeline("pipeline_advanced_ann", "en")
        return AdvancedPipeline.model_ann
