import sparknlp.internal as _internal
from pyspark.ml.param import *
import re

def RegexRule(rule, identifier):
    return _internal._RegexRule(rule, identifier).apply()

def ExternalResource(path, read_as, options):
    return _internal._ExternalResource(path, read_as, options).apply()

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
