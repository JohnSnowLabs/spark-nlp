from pyspark import SparkContext
from pyspark.ml.wrapper import JavaWrapper


class ExtendedJavaWrapper(JavaWrapper):
    def __init__(self, java_obj):
        super(ExtendedJavaWrapper, self).__init__(java_obj)
        self.sc = SparkContext._active_spark_context
        self.java_obj = self._java_obj

    def new_java_obj(self, java_class, *args):
        return self._new_java_obj(java_class, *args)

    def new_java_array(self, pylist, java_class):
        """
        ToDo: Inspired from spark 2.2.0. Delete if we upgrade
        """
        java_array = self.sc._gateway.new_array(java_class, len(pylist))
        for i in range(len(pylist)):
            java_array[i] = pylist[i]
        return java_array


class RegexRule(ExtendedJavaWrapper):
    def __init__(self, rule, identifier):
        super(RegexRule, self).__init__("com.johnsnowlabs.nlp.util.regex.RegexRule")
        self._java_obj = self._new_java_obj(self._java_obj, rule, identifier)

    def __call__(self):
        return self._java_obj


class TaggedWord(ExtendedJavaWrapper):
    def __init__(self, word, tag):
        super(TaggedWord, self).__init__("com.johnsnowlabs.nlp.annotators.common.TaggedWord")
        self._java_obj = self._new_java_obj(self._java_obj, word, tag)

    def __call__(self):
        return self._java_obj


class TaggedSentence(ExtendedJavaWrapper):
    def __init__(self, tagged_words):
        super(TaggedSentence, self).__init__("com.johnsnowlabs.nlp.annotators.common.TaggedSentence")
        self._java_obj = self._new_java_obj(self._java_obj, tagged_words)

    def __call__(self):
        return self._java_obj


class TokenizedSentence(ExtendedJavaWrapper):
    def __init__(self, tokens):
        super(TokenizedSentence, self).__init__("com.johnsnowlabs.nlp.annotators.common.TokenizedSentence")
        self._java_obj = self._new_java_obj(self._java_obj, tokens)

    def __call__(self):
        return self._java_obj


class ConfigLoaderGetter(ExtendedJavaWrapper):
    def __init__(self):
        super(ConfigLoaderGetter, self).__init__("com.johnsnowlabs.util.ConfigLoader.getConfigPath")
        self._java_obj = self._new_java_obj(self._java_obj)

    def __call__(self):
        return self._java_obj


class ConfigLoaderSetter(ExtendedJavaWrapper):
    def __init__(self, path):
        super(ConfigLoaderSetter, self).__init__("com.johnsnowlabs.util.ConfigLoader.setConfigPath")
        self._java_obj = self._new_java_obj(self._java_obj, path)

    def __call__(self):
        return self._java_obj
