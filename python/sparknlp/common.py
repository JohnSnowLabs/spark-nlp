from sparknlp.util import ExtendedJavaWrapper


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


