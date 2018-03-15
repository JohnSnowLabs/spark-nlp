import sparknlp.internal as _internal
from pyspark.ml.param import *
import re


def RegexRule(rule, identifier):
    return _internal._RegexRule(rule, identifier).apply()


def ExternalResource(path, read_as, options):
    return _internal._ExternalResource(path, read_as, options).apply()


class Annotation:
    def __init__(self, annotator_type, start, end, result, metadata):
        self.annotator_type = annotator_type
        self.start = start
        self.end = end
        self.result = result
        self.metadata = metadata

class SparklessPipeline:
    def __init__(self, pipelineModel):
        self._sparklessPipeline = _internal._SparklessPipeline(pipelineModel).apply()

    @staticmethod
    def _annotation_from_java(java_annotations):
        annotations = []
        for annotation in java_annotations:
            annotations.append(Annotation(annotation.annotatorType(),
                                          annotation.start(),
                                          annotation.end(),
                                          annotation.result(),
                                          dict(annotation.metadata()))
                               )
        return annotations

    def annotate(self, target):
        collected = self._sparklessPipeline.annotate(target)
        result = {}
        for atype, annotations in collected.items():
            result[atype] = self._annotation_from_java(annotations)
        return result

"""
Helper class used to generate the getters for all params
"""
class ParamsGetters(Params):
    getter_attrs = []

    def __init__(self):
        super(ParamsGetters, self).__init__()
        for param in self.params:
            param_name = param.name
            f_attr = "get" + re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name)
            setattr(self, f_attr, self.getParamValue(param_name))

    def getParamValue(self, paramName):
        def r():
            try:
                return self.getOrDefault(paramName)
            except:
                return None
        return r
