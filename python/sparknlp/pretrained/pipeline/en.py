from sparknlp.pretrained import ResourceDownloader
from sparknlp.base import LightPipeline
import pyspark


class _Model:
    @staticmethod
    def annotate(model_class, name, target, target_column=None):
        if not model_class.model:
            model_class.model = ResourceDownloader().downloadPipeline(name, "en")
        if type(target) is pyspark.sql.dataframe.DataFrame:
            if not target_column:
                raise Exception("annotate() target_column arg needed when targeting a DataFrame")
            return model_class.model.transform(target.withColumnRenamed(target_column, "text"))
        elif type(target) is list or type(target) is str:
            pip = LightPipeline(model_class.model)
            return pip.annotate(target)


class BasicPipeline:
    model = None

    @staticmethod
    def annotate(target, target_column=None):
        return _Model.annotate(BasicPipeline, "pipeline_basic", target, target_column)

    @staticmethod
    def pretrained():
        if not BasicPipeline.model:
            BasicPipeline.model = ResourceDownloader().downloadPipeline("pipeline_basic", "en")
        return BasicPipeline.model


class AdvancedPipeline:
    model = None

    @staticmethod
    def annotate(target, target_column=None):
        return _Model.annotate(AdvancedPipeline, "pipeline_advanced", target, target_column)

    @staticmethod
    def pretrained():
        if not BasicPipeline.model:
            AdvancedPipeline.model = ResourceDownloader().downloadPipeline("pipeline_advanced", "en")
        return AdvancedPipeline.model


class SentimentPipeline:
    model = None

    @staticmethod
    def annotate(target, target_column=None):
        return _Model.annotate(AdvancedPipeline, "pipeline_vivekn", target, target_column)

    @staticmethod
    def pretrained():
        if not BasicPipeline.model:
            AdvancedPipeline.model = ResourceDownloader().downloadPipeline("pipeline_vivekn", "en")
        return AdvancedPipeline.model
