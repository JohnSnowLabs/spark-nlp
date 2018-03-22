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


class LightPipeline:
    def __init__(self, pipelineModel):
        self._lightPipeline = _internal._LightPipeline(pipelineModel).apply()

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

    def fullAnnotate(self, target):
        result = []
        for row in self._lightPipeline.fullAnnotateJava(target):
            kas = {}
            for atype, annotations in row.items():
                kas[atype] = self._annotation_from_java(annotations)
            result.append(kas)
        return result

    def annotate(self, target):
        def extract(text_annotations):
            kas = {}
            for atype, text in text_annotations.items():
                kas[atype] = text
            return kas

        annotations = self._lightPipeline.annotateJava(target)

        if type(target) is str:
            result = extract(annotations)
        elif type(target) is list:
            result = []
            for row_annotations in annotations:
                result.append(extract(row_annotations))
        else:
            raise TypeError("target for annotation may be 'str' or 'list'")

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
