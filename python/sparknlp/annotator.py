##
# Prototyping for py4j to pipeline from Python
##

import sys
from sparknlp.common import *

# Do NOT delete. Looks redundant but this is key work around for python 2 support.
if sys.version_info[0] == 2:
    from sparknlp.base import DocumentAssembler, Finisher, EmbeddingsFinisher, TokenAssembler
else:
    import com.johnsnowlabs.nlp

annotators = sys.modules[__name__]
pos = sys.modules[__name__]
pos.perceptron = sys.modules[__name__]
ner = sys.modules[__name__]
ner.crf = sys.modules[__name__]
ner.dl = sys.modules[__name__]
regex = sys.modules[__name__]
sbd = sys.modules[__name__]
sbd.pragmatic = sys.modules[__name__]
sda = sys.modules[__name__]
sda.pragmatic = sys.modules[__name__]
sda.vivekn = sys.modules[__name__]
spell = sys.modules[__name__]
spell.norvig = sys.modules[__name__]
spell.symmetric = sys.modules[__name__]
spell.context = sys.modules[__name__]
parser = sys.modules[__name__]
parser.dep = sys.modules[__name__]
parser.typdep = sys.modules[__name__]
embeddings = sys.modules[__name__]
classifier = sys.modules[__name__]
classifier.dl = sys.modules[__name__]
ld = sys.modules[__name__]
ld.dl = sys.modules[__name__]
keyword = sys.modules[__name__]
keyword.yake = sys.modules[__name__]
sentence_detector_dl = sys.modules[__name__]
seq2seq = sys.modules[__name__]
ws = sys.modules[__name__]


class RecursiveTokenizer(AnnotatorApproach):
    name = 'RecursiveTokenizer'

    prefixes = Param(Params._dummy(),
                     "prefixes",
                     "strings to be considered independent tokens when found at the beginning of a word",
                     typeConverter=TypeConverters.toListString)

    suffixes = Param(Params._dummy(),
                     "suffixes",
                     "strings to be considered independent tokens when found at the end of a word",
                     typeConverter=TypeConverters.toListString)

    infixes = Param(Params._dummy(),
                    "infixes",
                    "strings to be considered independent tokens when found in the middle of a word",
                    typeConverter=TypeConverters.toListString)

    whitelist = Param(Params._dummy(),
                      "whitelist",
                      "strings to be considered as single tokens",
                      typeConverter=TypeConverters.toListString)

    def setPrefixes(self, p):
        return self._set(prefixes=p)

    def setSuffixes(self, s):
        return self._set(suffixes=s)

    def setInfixes(self, i):
        return self._set(infixes=i)

    def setWhitelist(self, w):
        return self._set(whitelist=w)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.RecursiveTokenizer"):
        super(RecursiveTokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.RecursiveTokenizer")
        self._setDefault(
            prefixes=["'", "\"", "(", "[", "\n"],
            infixes=["\n", "(", ")"],
            suffixes=[".", ":", "%", ",", ";", "?", "'", "\"", ")", "]", "\n", "!", "'s"],
            whitelist=["it's", "that's", "there's", "he's", "she's", "what's", "let's", "who's", \
                       "It's", "That's", "There's", "He's", "She's", "What's", "Let's", "Who's"]
        )


    def _create_model(self, java_model):
        return RecursiveTokenizerModel(java_model=java_model)


class RecursiveTokenizerModel(AnnotatorModel):
    name = 'RecursiveTokenizerModel'

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.RecursiveTokenizerModel", java_model=None):
        super(RecursiveTokenizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )


class Tokenizer(AnnotatorApproach):

    targetPattern = Param(Params._dummy(),
                          "targetPattern",
                          "pattern to grab from text as token candidates. Defaults \S+",
                          typeConverter=TypeConverters.toString)

    prefixPattern = Param(Params._dummy(),
                          "prefixPattern",
                          "regex with groups and begins with \A to match target prefix. Defaults to \A([^\s\w\$\.]*)",
                          typeConverter=TypeConverters.toString)

    suffixPattern = Param(Params._dummy(),
                          "suffixPattern",
                          "regex with groups and ends with \z to match target suffix. Defaults to ([^\s\w]?)([^\s\w]*)\z",
                          typeConverter=TypeConverters.toString)

    infixPatterns = Param(Params._dummy(),
                          "infixPatterns",
                          "regex patterns that match tokens within a single target. groups identify different sub-tokens. multiple defaults",
                          typeConverter=TypeConverters.toListString)

    exceptions = Param(Params._dummy(),
                       "exceptions",
                       "Words that won't be affected by tokenization rules",
                       typeConverter=TypeConverters.toListString)

    exceptionsPath = Param(Params._dummy(),
                           "exceptionsPath",
                           "path to file containing list of exceptions",
                           typeConverter=TypeConverters.toString)

    caseSensitiveExceptions = Param(Params._dummy(),
                                    "caseSensitiveExceptions",
                                    "Whether to care for case sensitiveness in exceptions",
                                    typeConverter=TypeConverters.toBoolean)

    contextChars = Param(Params._dummy(),
                         "contextChars",
                         "character list used to separate from token boundaries",
                         typeConverter=TypeConverters.toListString)

    splitPattern = Param(Params._dummy(),
                         "splitPattern",
                         "character list used to separate from the inside of tokens",
                         typeConverter=TypeConverters.toString)

    splitChars = Param(Params._dummy(),
                       "splitChars",
                       "character list used to separate from the inside of tokens",
                       typeConverter=TypeConverters.toListString)

    minLength = Param(Params._dummy(),
                      "minLength",
                      "Set the minimum allowed legth for each token",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "Set the maximum allowed legth for each token",
                      typeConverter=TypeConverters.toInt)

    name = 'Tokenizer'

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Tokenizer")
        self._setDefault(
            targetPattern="\\S+",
            contextChars=[".", ",", ";", ":", "!", "?", "*", "-", "(", ")", "\"", "'"],
            caseSensitiveExceptions=True,
            minLength=0,
            maxLength=99999
        )

    def getInfixPatterns(self):
        return self.getOrDefault("infixPatterns")

    def getSuffixPattern(self):
        return self.getOrDefault("suffixPattern")

    def getPrefixPattern(self):
        return self.getOrDefault("prefixPattern")

    def getContextChars(self):
        return self.getOrDefault("contextChars")

    def getSplitChars(self):
        return self.getOrDefault("splitChars")

    def setTargetPattern(self, value):
        return self._set(targetPattern=value)

    def setPrefixPattern(self, value):
        return self._set(prefixPattern=value)

    def setSuffixPattern(self, value):
        return self._set(suffixPattern=value)

    def setInfixPatterns(self, value):
        return self._set(infixPatterns=value)

    def addInfixPattern(self, value):
        try:
            infix_patterns = self.getInfixPatterns()
        except KeyError:
            infix_patterns = []
        infix_patterns.insert(0, value)
        return self._set(infixPatterns=infix_patterns)

    def setExceptions(self, value):
        return self._set(exceptions=value)

    def getExceptions(self):
        return self.getOrDefault("exceptions")

    def addException(self, value):
        try:
            exception_tokens = self.getExceptions()
        except KeyError:
            exception_tokens = []
        exception_tokens.append(value)
        return self._set(exceptions=exception_tokens)

    def setCaseSensitiveExceptions(self, value):
        return self._set(caseSensitiveExceptions=value)

    def getCaseSensitiveExceptions(self):
        return self.getOrDefault("caseSensitiveExceptions")

    def setContextChars(self, value):
        return self._set(contextChars=value)

    def addContextChars(self, value):
        try:
            context_chars = self.getContextChars()
        except KeyError:
            context_chars = []
        context_chars.append(value)
        return self._set(contextChars=context_chars)

    def setSplitPattern(self, value):
        return self._set(splitPattern=value)

    def setSplitChars(self, value):
        return self._set(splitChars=value)

    def addSplitChars(self, value):
        try:
            split_chars = self.getSplitChars()
        except KeyError:
            split_chars = []
        split_chars.append(value)
        return self._set(splitChars=split_chars)

    def setMinLength(self, value):
        return self._set(minLength=value)

    def setMaxLength(self, value):
        return self._set(maxLength=value)

    def _create_model(self, java_model):
        return TokenizerModel(java_model=java_model)


class TokenizerModel(AnnotatorModel):
    name = "TokenizerModel"

    exceptions = Param(Params._dummy(),
                       "exceptions",
                       "Words that won't be affected by tokenization rules",
                       typeConverter=TypeConverters.toListString)

    caseSensitiveExceptions = Param(Params._dummy(),
                                    "caseSensitiveExceptions",
                                    "Whether to care for case sensitiveness in exceptions",
                                    typeConverter=TypeConverters.toBoolean)

    targetPattern = Param(Params._dummy(),
                          "targetPattern",
                          "pattern to grab from text as token candidates. Defaults \S+",
                          typeConverter=TypeConverters.toString)

    rules = Param(Params._dummy(),
                  "rules",
                  "Rules structure factory containing pre processed regex rules",
                  typeConverter=TypeConverters.identity)

    splitPattern = Param(Params._dummy(),
                         "splitPattern",
                         "character list used to separate from the inside of tokens",
                         typeConverter=TypeConverters.toString)

    splitChars = Param(Params._dummy(),
                       "splitChars",
                       "character list used to separate from the inside of tokens",
                       typeConverter=TypeConverters.toListString)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.TokenizerModel", java_model=None):
        super(TokenizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            targetPattern="\\S+",
            caseSensitiveExceptions=True
        )

    def setSplitPattern(self, value):
        return self._set(splitPattern=value)

    def setSplitChars(self, value):
        return self._set(splitChars=value)

    def addSplitChars(self, value):
        try:
            split_chars = self.getSplitChars()
        except KeyError:
            split_chars = []
        split_chars.append(value)
        return self._set(splitChars=split_chars)

    @staticmethod
    def pretrained(name="token_rules", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TokenizerModel, name, lang, remote_loc)


class RegexTokenizer(AnnotatorModel):

    name = "RegexTokenizer"

    @keyword_only
    def __init__(self):
        super(RegexTokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.RegexTokenizer")
        self._setDefault(
            inputCols=["document"],
            outputCol="regexToken",
            toLowercase=False,
            minLength=1,
            pattern="\\s+",
            positionalMask=False
        )

    minLength = Param(Params._dummy(),
                      "minLength",
                      "Set the minimum allowed legth for each token",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "Set the maximum allowed legth for each token",
                      typeConverter=TypeConverters.toInt)

    toLowercase = Param(Params._dummy(),
                        "toLowercase",
                        "Indicates whether to convert all characters to lowercase before tokenizing.",
                        typeConverter=TypeConverters.toBoolean)

    pattern = Param(Params._dummy(),
                    "pattern",
                    "regex pattern used for tokenizing. Defaults \S+",
                    typeConverter=TypeConverters.toString)

    positionalMask = Param(Params._dummy(),
                           "positionalMask",
                           "Using a positional mask to guarantee the incremental progression of the tokenization.",
                           typeConverter=TypeConverters.toBoolean)

    def setMinLength(self, value):
        return self._set(minLength=value)

    def setMaxLength(self, value):
        return self._set(maxLength=value)

    def setToLowercase(self, value):
        return self._set(toLowercase=value)

    def setPattern(self, value):
        return self._set(pattern=value)

    def setPositionalMask(self, value):
        return self._set(positionalMask=value)


class ChunkTokenizer(Tokenizer):
    name = 'ChunkTokenizer'

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ChunkTokenizer")

    def _create_model(self, java_model):
        return ChunkTokenizerModel(java_model=java_model)


class ChunkTokenizerModel(TokenizerModel):
    name = 'ChunkTokenizerModel'

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ChunkTokenizerModel", java_model=None):
        super(TokenizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )


class Token2Chunk(AnnotatorModel):
    name = "Token2Chunk"

    def __init__(self):
        super(Token2Chunk, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Token2Chunk")


class Stemmer(AnnotatorModel):

    language = Param(Params._dummy(), "language", "stemmer algorithm", typeConverter=TypeConverters.toString)

    name = "Stemmer"

    @keyword_only
    def __init__(self):
        super(Stemmer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Stemmer")
        self._setDefault(
            language="english"
        )


class Chunker(AnnotatorModel):

    regexParsers = Param(Params._dummy(),
                         "regexParsers",
                         "an array of grammar based chunk parsers",
                         typeConverter=TypeConverters.toListString)

    name = "Chunker"

    @keyword_only
    def __init__(self):
        super(Chunker, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Chunker")

    def setRegexParsers(self, value):
        return self._set(regexParsers=value)


class DocumentNormalizer(AnnotatorModel):

    action = Param(Params._dummy(),
                   "action",
                   "action to perform applying regex patterns on text",
                   typeConverter=TypeConverters.toString)

    patterns = Param(Params._dummy(),
                     "patterns",
                     "normalization regex patterns which match will be removed from document. Defaults is <[^>]*>",
                     typeConverter=TypeConverters.toListString)

    replacement = Param(Params._dummy(),
                        "replacement",
                        "replacement string to apply when regexes match",
                        typeConverter=TypeConverters.toString)

    lowercase = Param(Params._dummy(),
                      "lowercase",
                      "whether to convert strings to lowercase",
                      typeConverter=TypeConverters.toBoolean)

    policy = Param(Params._dummy(),
                   "policy",
                   "policy to remove pattern from text",
                   typeConverter=TypeConverters.toString)

    encoding = Param(Params._dummy(),
                     "encoding",
                     "file encoding to apply on normalized documents",
                     typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(DocumentNormalizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.DocumentNormalizer")
        self._setDefault(
            action="clean",
            patterns=["<[^>]*>"],
            replacement=" ",
            lowercase=False,
            policy="pretty_all",
            encoding="UTF-8"
        )

    def setAction(self, value):
        return self._set(action=value)

    def setPatterns(self, value):
        return self._set(patterns=value)

    def setReplacement(self, value):
        return self._set(replacement=value)

    def setLowercase(self, value):
        return self._set(lowercase=value)

    def setPolicy(self, value):
        return self._set(policy=value)

    def setEncoding(self, value):
        return self._set(encoding=value)


class Normalizer(AnnotatorApproach):

    cleanupPatterns = Param(Params._dummy(),
                            "cleanupPatterns",
                            "normalization regex patterns which match will be removed from token",
                            typeConverter=TypeConverters.toListString)

    lowercase = Param(Params._dummy(),
                      "lowercase",
                      "whether to convert strings to lowercase")

    slangMatchCase = Param(Params._dummy(),
                           "slangMatchCase",
                           "whether or not to be case sensitive to match slangs. Defaults to false.")

    slangDictionary = Param(Params._dummy(),
                            "slangDictionary",
                            "slang dictionary is a delimited text. needs 'delimiter' in options",
                            typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(Normalizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Normalizer")
        self._setDefault(
            cleanupPatterns=["[^\\pL+]"],
            lowercase=False,
            slangMatchCase=False
        )

    def setCleanupPatterns(self, value):
        return self._set(cleanupPatterns=value)

    def setLowercase(self, value):
        return self._set(lowercase=value)

    def setSlangDictionary(self, path, delimiter, read_as=ReadAs.TEXT, options={"format": "text"}):
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(slangDictionary=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return NormalizerModel(java_model=java_model)


class NormalizerModel(AnnotatorModel):

    cleanupPatterns = Param(Params._dummy(),
                            "cleanupPatterns",
                            "normalization regex patterns which match will be removed from token",
                            typeConverter=TypeConverters.toListString)

    lowercase = Param(Params._dummy(),
                      "lowercase",
                      "whether to convert strings to lowercase")

    slangMatchCase = Param(Params._dummy(),
                           "slangMatchCase",
                           "whether or not to be case sensitive to match slangs. Defaults to false.")

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.NormalizerModel", java_model=None):
        super(NormalizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    name = "NormalizerModel"


class RegexMatcher(AnnotatorApproach):

    strategy = Param(Params._dummy(),
                     "strategy",
                     "MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE",
                     typeConverter=TypeConverters.toString)
    externalRules = Param(Params._dummy(),
                          "externalRules",
                          "external resource to rules, needs 'delimiter' in options",
                          typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(RegexMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.RegexMatcher")
        self._setDefault(
            strategy="MATCH_ALL"
        )

    def setStrategy(self, value):
        return self._set(strategy=value)

    def setExternalRules(self, path, delimiter, read_as=ReadAs.TEXT, options={"format": "text"}):
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(externalRules=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return RegexMatcherModel(java_model=java_model)


class RegexMatcherModel(AnnotatorModel):
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.RegexMatcherModel", java_model=None):
        super(RegexMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    name = "RegexMatcherModel"


class Lemmatizer(AnnotatorApproach):
    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "lemmatizer external dictionary." +
                       " needs 'keyDelimiter' and 'valueDelimiter' in options for parsing target text",
                       typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(Lemmatizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Lemmatizer")

    def _create_model(self, java_model):
        return LemmatizerModel(java_model=java_model)

    def setDictionary(self, path, key_delimiter, value_delimiter, read_as=ReadAs.TEXT,
                      options={"format": "text"}):
        opts = options.copy()
        if "keyDelimiter" not in opts:
            opts["keyDelimiter"] = key_delimiter
        if "valueDelimiter" not in opts:
            opts["valueDelimiter"] = value_delimiter
        return self._set(dictionary=ExternalResource(path, read_as, opts))


class LemmatizerModel(AnnotatorModel):
    name = "LemmatizerModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.LemmatizerModel", java_model=None):
        super(LemmatizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="lemma_antbnc", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(LemmatizerModel, name, lang, remote_loc)


class DateMatcherUtils(Params):
    dateFormat = Param(Params._dummy(),
                       "dateFormat",
                       "desired format for dates extracted",
                       typeConverter=TypeConverters.toString)

    readMonthFirst = Param(Params._dummy(),
                           "readMonthFirst",
                           "Whether to parse july 07/05/2015 or as 05/07/2015",
                           typeConverter=TypeConverters.toBoolean
                           )

    defaultDayWhenMissing = Param(Params._dummy(),
                                  "defaultDayWhenMissing",
                                  "which day to set when it is missing from parsed input",
                                  typeConverter=TypeConverters.toInt
                                  )

    anchorDateYear = Param(Params._dummy(),
                           "anchorDateYear",
                           "Add an anchor year for the relative dates such as a day after tomorrow. If not set it "
                           "will use the current year. Example: 2021",
                           typeConverter=TypeConverters.toInt
                           )

    anchorDateMonth = Param(Params._dummy(),
                            "anchorDateMonth",
                            "Add an anchor month for the relative dates such as a day after tomorrow. If not set it "
                            "will use the current month. Example: 1 which means January",
                            typeConverter=TypeConverters.toInt
                            )

    anchorDateDay = Param(Params._dummy(),
                          "anchorDateDay",
                          "Add an anchor day of the day for the relative dates such as a day after tomorrow. If not "
                          "set it will use the current day. Example: 11",
                          typeConverter=TypeConverters.toInt
                          )

    def setFormat(self, value):
        return self._set(dateFormat=value)

    def setReadMonthFirst(self, value):
        return self._set(readMonthFirst=value)

    def setDefaultDayWhenMissing(self, value):
        return self._set(defaultDayWhenMissing=value)

    def setAnchorDateYear(self, value):
        return self._set(anchorDateYear=value)

    def setAnchorDateMonth(self, value):
        return self._set(anchorDateMonth=value)

    def setAnchorDateDay(self, value):
        return self._set(anchorDateDay=value)


class DateMatcher(AnnotatorModel, DateMatcherUtils):

    name = "DateMatcher"

    @keyword_only
    def __init__(self):
        super(DateMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.DateMatcher")
        self._setDefault(
            dateFormat="yyyy/MM/dd",
            readMonthFirst=True,
            defaultDayWhenMissing=1,
            anchorDateYear=-1,
            anchorDateMonth=-1,
            anchorDateDay=-1
        )


class MultiDateMatcher(AnnotatorModel, DateMatcherUtils):

    name = "MultiDateMatcher"

    @keyword_only
    def __init__(self):
        super(MultiDateMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.MultiDateMatcher")
        self._setDefault(
            dateFormat="yyyy/MM/dd",
            readMonthFirst=True,
            defaultDayWhenMissing=1
        )


class TextMatcher(AnnotatorApproach):

    entities = Param(Params._dummy(),
                     "entities",
                     "ExternalResource for entities",
                     typeConverter=TypeConverters.identity)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to match regardless of case. Defaults true",
                          typeConverter=TypeConverters.toBoolean)

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    entityValue = Param(Params._dummy(),
                        "entityValue",
                        "value for the entity metadata field",
                        typeConverter=TypeConverters.toString)


    buildFromTokens = Param(Params._dummy(),
                            "buildFromTokens",
                            "whether the TextMatcher should take the CHUNK from TOKEN or not",
                            typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(TextMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.TextMatcher")
        self._setDefault(caseSensitive=True)
        self._setDefault(mergeOverlapping=False)

    def _create_model(self, java_model):
        return TextMatcherModel(java_model=java_model)

    def setEntities(self, path, read_as=ReadAs.TEXT, options={"format": "text"}):
        return self._set(entities=ExternalResource(path, read_as, options.copy()))

    def setCaseSensitive(self, b):
        return self._set(caseSensitive=b)

    def setMergeOverlapping(self, b):
        return self._set(mergeOverlapping=b)

    def setEntityValue(self, b):
        return self._set(entityValue=b)

    def setBuildFromTokens(self, b):
        return self._set(buildFromTokens=b)


class TextMatcherModel(AnnotatorModel):
    name = "TextMatcherModel"

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    searchTrie = Param(Params._dummy(),
                       "searchTrie",
                       "searchTrie",
                       typeConverter=TypeConverters.identity)

    entityValue = Param(Params._dummy(),
                        "entityValue",
                        "value for the entity metadata field",
                        typeConverter=TypeConverters.toString)


    buildFromTokens = Param(Params._dummy(),
                            "buildFromTokens",
                            "whether the TextMatcher should take the CHUNK from TOKEN or not",
                            typeConverter=TypeConverters.toBoolean)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.TextMatcherModel", java_model=None):
        super(TextMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    def setMergeOverlapping(self, b):
        return self._set(mergeOverlapping=b)

    def setEntityValue(self, b):
        return self._set(entityValue=b)

    def setBuildFromTokens(self, b):
        return self._set(buildFromTokens=b)

    @staticmethod
    def pretrained(name, lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TextMatcherModel, name, lang, remote_loc)


class BigTextMatcher(AnnotatorApproach, HasStorage):

    entities = Param(Params._dummy(),
                     "entities",
                     "ExternalResource for entities",
                     typeConverter=TypeConverters.identity)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case in index lookups",
                          typeConverter=TypeConverters.toBoolean)

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    tokenizer = Param(Params._dummy(),
                      "tokenizer",
                      "TokenizerModel to use to tokenize input file for building a Trie",
                      typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(BigTextMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.btm.BigTextMatcher")
        self._setDefault(caseSensitive=True)
        self._setDefault(mergeOverlapping=False)

    def _create_model(self, java_model):
        return TextMatcherModel(java_model=java_model)

    def setEntities(self, path, read_as=ReadAs.TEXT, options={"format": "text"}):
        return self._set(entities=ExternalResource(path, read_as, options.copy()))

    def setCaseSensitive(self, b):
        return self._set(caseSensitive=b)

    def setMergeOverlapping(self, b):
        return self._set(mergeOverlapping=b)

    def setTokenizer(self, tokenizer_model):
        tokenizer_model._transfer_params_to_java()
        return self._set(tokenizer_model._java_obj)


class BigTextMatcherModel(AnnotatorModel, HasStorageModel):
    name = "BigTextMatcherModel"
    databases = ['TMVOCAB', 'TMEDGES', 'TMNODES']

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case in index lookups",
                          typeConverter=TypeConverters.toBoolean)

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    searchTrie = Param(Params._dummy(),
                       "searchTrie",
                       "searchTrie",
                       typeConverter=TypeConverters.identity)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.btm.TextMatcherModel", java_model=None):
        super(BigTextMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    def setMergeOverlapping(self, b):
        return self._set(mergeOverlapping=b)

    def setCaseSensitive(self, v):
        return self._set(caseSensitive=v)

    @staticmethod
    def pretrained(name, lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TextMatcherModel, name, lang, remote_loc)

    @staticmethod
    def loadStorage(path, spark, storage_ref):
        HasStorageModel.loadStorages(path, spark, storage_ref, BigTextMatcherModel.databases)


class PerceptronApproach(AnnotatorApproach):
    posCol = Param(Params._dummy(),
                   "posCol",
                   "column of Array of POS tags that match tokens",
                   typeConverter=TypeConverters.toString)

    nIterations = Param(Params._dummy(),
                        "nIterations",
                        "Number of iterations in training, converges to better accuracy",
                        typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(PerceptronApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach")
        self._setDefault(
            nIterations=5
        )

    def setPosCol(self, value):
        return self._set(posCol=value)

    def setIterations(self, value):
        return self._set(nIterations=value)

    def getNIterations(self):
        return self.getOrDefault(self.nIterations)

    def _create_model(self, java_model):
        return PerceptronModel(java_model=java_model)


class PerceptronModel(AnnotatorModel):
    name = "PerceptronModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel", java_model=None):
        super(PerceptronModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="pos_anc", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(PerceptronModel, name, lang, remote_loc)


class SentenceDetectorParams:
    useAbbreviations = Param(Params._dummy(),
                             "useAbbreviations",
                             "whether to apply abbreviations at sentence detection",
                             typeConverter=TypeConverters.toBoolean)

    customBounds = Param(Params._dummy(),
                         "customBounds",
                         "characters used to explicitly mark sentence bounds",
                         typeConverter=TypeConverters.toListString)

    useCustomBoundsOnly = Param(Params._dummy(),
                                "useCustomBoundsOnly",
                                "Only utilize custom bounds in sentence detection",
                                typeConverter=TypeConverters.toBoolean)

    explodeSentences = Param(Params._dummy(),
                             "explodeSentences",
                             "whether to explode each sentence into a different row, for better parallelization. Defaults to false.",
                             typeConverter=TypeConverters.toBoolean)

    splitLength = Param(Params._dummy(),
                        "splitLength",
                        "length at which sentences will be forcibly split.",
                        typeConverter=TypeConverters.toInt)

    minLength = Param(Params._dummy(),
                      "minLength",
                      "Set the minimum allowed length for each sentence.",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "Set the maximum allowed length for each sentence",
                      typeConverter=TypeConverters.toInt)


class SentenceDetector(AnnotatorModel, SentenceDetectorParams):

    name = 'SentenceDetector'

    # this one is exclusive to this detector
    detectLists = Param(Params._dummy(),
                        "detectLists",
                        "whether detect lists during sentence detection",
                        typeConverter=TypeConverters.toBoolean)

    def setCustomBounds(self, value):
        return self._set(customBounds=value)

    def setUseAbbreviations(self, value):
        return self._set(useAbbreviations=value)

    def setDetectLists(self, value):
        return self._set(detectLists=value)

    def setUseCustomBoundsOnly(self, value):
        return self._set(useCustomBoundsOnly=value)

    def setExplodeSentences(self, value):
        return self._set(explodeSentences=value)

    def setSplitLength(self, value):
        return self._set(splitLength=value)

    def setMinLength(self, value):
        return self._set(minLength=value)

    def setMaxLength(self, value):
        return self._set(maxLength=value)

    @keyword_only
    def __init__(self):
        super(SentenceDetector, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector")
        self._setDefault(
            useAbbreviations=True,
            detectLists=True,
            useCustomBoundsOnly=False,
            customBounds=[],
            explodeSentences=False,
            minLength=0,
            maxLength=99999
        )


class SentimentDetector(AnnotatorApproach):
    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "path for dictionary to sentiment analysis",
                       typeConverter=TypeConverters.identity)

    positiveMultiplier = Param(Params._dummy(),
                               "positiveMultiplier",
                               "multiplier for positive sentiments. Defaults 1.0",
                               typeConverter=TypeConverters.toFloat)

    negativeMultiplier = Param(Params._dummy(),
                               "negativeMultiplier",
                               "multiplier for negative sentiments. Defaults -1.0",
                               typeConverter=TypeConverters.toFloat)

    incrementMultiplier = Param(Params._dummy(),
                                "incrementMultiplier",
                                "multiplier for increment sentiments. Defaults 2.0",
                                typeConverter=TypeConverters.toFloat)

    decrementMultiplier = Param(Params._dummy(),
                                "decrementMultiplier",
                                "multiplier for decrement sentiments. Defaults -2.0",
                                typeConverter=TypeConverters.toFloat)

    reverseMultiplier = Param(Params._dummy(),
                              "reverseMultiplier",
                              "multiplier for revert sentiments. Defaults -1.0",
                              typeConverter=TypeConverters.toFloat)

    enableScore = Param(Params._dummy(),
                        "enableScore",
                        "if true, score will show as the double value, else will output string \"positive\" or \"negative\". Defaults false",
                        typeConverter=TypeConverters.toBoolean)


    def __init__(self):
        super(SentimentDetector, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector")
        self._setDefault(positiveMultiplier=1.0, negativeMultiplier=-1.0, incrementMultiplier=2.0,
                         decrementMultiplier=-2.0, reverseMultiplier=-1.0, enableScore=False)

    def setDictionary(self, path, delimiter, read_as=ReadAs.TEXT, options={'format': 'text'}):
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return SentimentDetectorModel(java_model=java_model)


class SentimentDetectorModel(AnnotatorModel):
    name = "SentimentDetectorModel"

    positiveMultiplier = Param(Params._dummy(),
                               "positiveMultiplier",
                               "multiplier for positive sentiments. Defaults 1.0",
                               typeConverter=TypeConverters.toFloat)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel",
                 java_model=None):
        super(SentimentDetectorModel, self).__init__(
            classname=classname,
            java_model=java_model
        )


class ViveknSentimentApproach(AnnotatorApproach):
    sentimentCol = Param(Params._dummy(),
                         "sentimentCol",
                         "column with the sentiment result of every row. Must be 'positive' or 'negative'",
                         typeConverter=TypeConverters.toString)

    pruneCorpus = Param(Params._dummy(),
                        "pruneCorpus",
                        "Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1",
                        typeConverter=TypeConverters.toInt)

    importantFeatureRatio = Param(Params._dummy(),
                                  "importantFeatureRatio",
                                  "proportion of feature content to be considered relevant. Defaults to 0.5",
                                  typeConverter=TypeConverters.toFloat)

    unimportantFeatureStep = Param(Params._dummy(),
                                   "unimportantFeatureStep",
                                   "proportion to lookahead in unimportant features. Defaults to 0.025",
                                   typeConverter=TypeConverters.toFloat)

    featureLimit = Param(Params._dummy(),
                         "featureLimit",
                         "content feature limit, to boost performance in very dirt text. Default disabled with -1",
                         typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(ViveknSentimentApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach")
        self._setDefault(pruneCorpus=1, importantFeatureRatio=0.5, unimportantFeatureStep=0.025, featureLimit=-1)

    def setSentimentCol(self, value):
        return self._set(sentimentCol=value)

    def setPruneCorpus(self, value):
        return self._set(pruneCorpus=value)

    def _create_model(self, java_model):
        return ViveknSentimentModel(java_model=java_model)


class ViveknSentimentModel(AnnotatorModel):
    name = "ViveknSentimentModel"

    importantFeatureRatio = Param(Params._dummy(),
                                  "importantFeatureRatio",
                                  "proportion of feature content to be considered relevant. Defaults to 0.5",
                                  typeConverter=TypeConverters.toFloat)

    unimportantFeatureStep = Param(Params._dummy(),
                                   "unimportantFeatureStep",
                                   "proportion to lookahead in unimportant features. Defaults to 0.025",
                                   typeConverter=TypeConverters.toFloat)

    featureLimit = Param(Params._dummy(),
                         "featureLimit",
                         "content feature limit, to boost performance in very dirt text. Default disabled with -1",
                         typeConverter=TypeConverters.toInt)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel", java_model=None):
        super(ViveknSentimentModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="sentiment_vivekn", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ViveknSentimentModel, name, lang, remote_loc)


class NorvigSweetingApproach(AnnotatorApproach):
    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "dictionary needs 'tokenPattern' regex in dictionary for separating words",
                       typeConverter=TypeConverters.identity)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case sensitivty",
                          typeConverter=TypeConverters.toBoolean)

    doubleVariants = Param(Params._dummy(),
                           "doubleVariants",
                           "whether to use more expensive spell checker",
                           typeConverter=TypeConverters.toBoolean)

    shortCircuit = Param(Params._dummy(),
                         "shortCircuit",
                         "whether to use faster mode",
                         typeConverter=TypeConverters.toBoolean)

    frequencyPriority = Param(Params._dummy(),
                              "frequencyPriority",
                              "applies frequency over hamming in intersections. When false hamming takes priority",
                              typeConverter=TypeConverters.toBoolean)

    wordSizeIgnore = Param(Params._dummy(),
                           "wordSizeIgnore",
                           "minimum size of word before ignoring. Defaults to 3",
                           typeConverter=TypeConverters.toInt)

    dupsLimit = Param(Params._dummy(),
                      "dupsLimit",
                      "maximum duplicate of characters in a word to consider. Defaults to 2",
                      typeConverter=TypeConverters.toInt)

    reductLimit = Param(Params._dummy(),
                        "reductLimit",
                        "word reductions limit. Defaults to 3",
                        typeConverter=TypeConverters.toInt)

    intersections = Param(Params._dummy(),
                          "intersections",
                          "hamming intersections to attempt. Defaults to 10",
                          typeConverter=TypeConverters.toInt)

    vowelSwapLimit = Param(Params._dummy(),
                           "vowelSwapLimit",
                           "vowel swap attempts. Defaults to 6",
                           typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(NorvigSweetingApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach")
        self._setDefault(caseSensitive=False, doubleVariants=False, shortCircuit=False, wordSizeIgnore=3, dupsLimit=2,
                         reductLimit=3, intersections=10, vowelSwapLimit=6, frequencyPriority=True)
        self.dictionary_path = ""

    def setDictionary(self, path, token_pattern="\S+", read_as=ReadAs.TEXT, options={"format": "text"}):
        self.dictionary_path = path
        opts = options.copy()
        if "tokenPattern" not in opts:
            opts["tokenPattern"] = token_pattern
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def setCaseSensitive(self, value):
        return self._set(caseSensitive=value)

    def setDoubleVariants(self, value):
        return self._set(doubleVariants=value)

    def setShortCircuit(self, value):
        return self._set(shortCircuit=value)

    def setFrequencyPriority(self, value):
        return self._set(frequencyPriority=value)

    def _create_model(self, java_model):
        return NorvigSweetingModel(java_model=java_model)


class NorvigSweetingModel(AnnotatorModel):
    name = "NorvigSweetingModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel", java_model=None):
        super(NorvigSweetingModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="spellcheck_norvig", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NorvigSweetingModel, name, lang, remote_loc)


class SymmetricDeleteApproach(AnnotatorApproach):
    corpus = Param(Params._dummy(),
                   "corpus",
                   "folder or file with text that teaches about the language",
                   typeConverter=TypeConverters.identity)

    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "folder or file with text that teaches about the language",
                       typeConverter=TypeConverters.identity)

    maxEditDistance = Param(Params._dummy(),
                            "maxEditDistance",
                            "max edit distance characters to derive strings from a word",
                            typeConverter=TypeConverters.toInt)

    frequencyThreshold = Param(Params._dummy(),
                               "frequencyThreshold",
                               "minimum frequency of words to be considered from training. " +
                               "Increase if training set is LARGE. Defaults to 0",
                               typeConverter=TypeConverters.toInt)

    deletesThreshold = Param(Params._dummy(),
                             "deletesThreshold",
                             "minimum frequency of corrections a word needs to have to be considered from training." +
                             "Increase if training set is LARGE. Defaults to 0",
                             typeConverter=TypeConverters.toInt)

    dupsLimit = Param(Params._dummy(),
                      "dupsLimit",
                      "maximum duplicate of characters in a word to consider. Defaults to 2",
                      typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(SymmetricDeleteApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach")
        self._setDefault(maxEditDistance=3, frequencyThreshold=0, deletesThreshold=0, dupsLimit=2)
        self.dictionary_path = ""

    def setDictionary(self, path, token_pattern="\S+", read_as=ReadAs.TEXT, options={"format": "text"}):
        self.dictionary_path = path
        opts = options.copy()
        if "tokenPattern" not in opts:
            opts["tokenPattern"] = token_pattern
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def setMaxEditDistance(self, v):
        return self._set(maxEditDistance=v)

    def setFrequencyThreshold(self, v):
        return self._set(frequencyThreshold=v)

    def setDeletesThreshold(self, v):
        return self._set(deletesThreshold=v)

    def _create_model(self, java_model):
        return SymmetricDeleteModel(java_model=java_model)


class SymmetricDeleteModel(AnnotatorModel):
    name = "SymmetricDeleteModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel",
                 java_model=None):
        super(SymmetricDeleteModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="spellcheck_sd", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SymmetricDeleteModel, name, lang, remote_loc)


class NerApproach(Params):
    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Column with label per each token",
                        typeConverter=TypeConverters.toString)

    entities = Param(Params._dummy(), "entities", "Entities to recognize", TypeConverters.toListString)

    minEpochs = Param(Params._dummy(), "minEpochs", "Minimum number of epochs to train", TypeConverters.toInt)
    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)
    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)

    def setLabelColumn(self, value):
        return self._set(labelColumn=value)

    def setEntities(self, tags):
        return self._set(entities=tags)

    def setMinEpochs(self, epochs):
        return self._set(minEpochs=epochs)

    def setMaxEpochs(self, epochs):
        return self._set(maxEpochs=epochs)

    def setVerbose(self, verboseValue):
        return self._set(verbose=verboseValue)

    def setRandomSeed(self, seed):
        return self._set(randomSeed=seed)

    def getLabelColumn(self):
        return self.getOrDefault(self.labelColumn)


class NerCrfApproach(AnnotatorApproach, NerApproach):

    l2 = Param(Params._dummy(), "l2", "L2 regularization coefficient", TypeConverters.toFloat)

    c0 = Param(Params._dummy(), "c0", "c0 params defining decay speed for gradient", TypeConverters.toInt)

    lossEps = Param(Params._dummy(), "lossEps", "If Epoch relative improvement less than eps then training is stopped",
                    TypeConverters.toFloat)

    minW = Param(Params._dummy(), "minW", "Features with less weights then this param value will be filtered",
                 TypeConverters.toFloat)

    includeConfidence = Param(Params._dummy(), "includeConfidence", "external features is a delimited text. needs 'delimiter' in options",
                              TypeConverters.toBoolean)

    externalFeatures = Param(Params._dummy(), "externalFeatures", "Additional dictionaries paths to use as a features",
                             TypeConverters.identity)

    def setL2(self, l2value):
        return self._set(l2=l2value)

    def setC0(self, c0value):
        return self._set(c0=c0value)

    def setLossEps(self, eps):
        return self._set(lossEps=eps)

    def setMinW(self, w):
        return self._set(minW=w)

    def setExternalFeatures(self, path, delimiter, read_as=ReadAs.TEXT, options={"format": "text"}):
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(externalFeatures=ExternalResource(path, read_as, opts))

    def setIncludeConfidence(self, b):
        return self._set(includeConfidence=b)

    def _create_model(self, java_model):
        return NerCrfModel(java_model=java_model)

    @keyword_only
    def __init__(self):
        super(NerCrfApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach")
        self._setDefault(
            minEpochs=0,
            maxEpochs=1000,
            l2=float(1),
            c0=2250000,
            lossEps=float(1e-3),
            verbose=4,
            includeConfidence=False
        )


class NerCrfModel(AnnotatorModel):
    name = "NerCrfModel"

    includeConfidence = Param(Params._dummy(), "includeConfidence", "external features is a delimited text. needs 'delimiter' in options",
                              TypeConverters.toBoolean)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel", java_model=None):
        super(NerCrfModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    def setIncludeConfidence(self, b):
        return self._set(includeConfidence=b)

    @staticmethod
    def pretrained(name="ner_crf", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NerCrfModel, name, lang, remote_loc)


class NerDLApproach(AnnotatorApproach, NerApproach):

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    po = Param(Params._dummy(), "po", "Learning rate decay coefficient. Real Learning Rage = lr / (1 + po * epoch)",
               TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)

    graphFolder = Param(Params._dummy(), "graphFolder", "Folder path that contain external graph files", TypeConverters.toString)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)

    useContrib = Param(Params._dummy(), "useContrib", "whether to use contrib LSTM Cells. Not compatible with Windows. Might slightly improve accuracy.", TypeConverters.toBoolean)

    validationSplit = Param(Params._dummy(), "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    evaluationLogExtended = Param(Params._dummy(), "evaluationLogExtended", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                                  TypeConverters.toBoolean)

    testDataset = Param(Params._dummy(), "testDataset",
                        "Path to test dataset. If set used to calculate statistic on it during training.",
                        TypeConverters.identity)

    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "whether to include confidence scores in annotation metadata",
                              TypeConverters.toBoolean)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

    outputLogsPath = Param(Params._dummy(), "outputLogsPath", "Folder path to save training logs", TypeConverters.toString)

    enableMemoryOptimizer = Param(Params._dummy(), "enableMemoryOptimizer", "Whether to optimize for large datasets or not. Enabling this option can slow down training.", TypeConverters.toBoolean)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setGraphFolder(self, p):
        return self._set(graphFolder=p)

    def setUseContrib(self, v):
        if v and sys.version == 'win32':
            raise Exception("Windows not supported to use contrib")
        return self._set(useContrib=v)

    def setLr(self, v):
        self._set(lr=v)
        return self

    def setPo(self, v):
        self._set(po=v)
        return self

    def setBatchSize(self, v):
        self._set(batchSize=v)
        return self

    def setDropout(self, v):
        self._set(dropout=v)
        return self

    def _create_model(self, java_model):
        return NerDLModel(java_model=java_model)

    def setValidationSplit(self, v):
        self._set(validationSplit=v)
        return self

    def setEvaluationLogExtended(self, v):
        self._set(evaluationLogExtended=v)
        return self

    def setTestDataset(self, path, read_as=ReadAs.SPARK, options={"format": "parquet"}):
        return self._set(testDataset=ExternalResource(path, read_as, options.copy()))

    def setIncludeConfidence(self, value):
        return self._set(includeConfidence=value)

    def setEnableOutputLogs(self, value):
        return self._set(enableOutputLogs=value)

    def setEnableMemoryOptimizer(self, value):
        return self._set(enableMemoryOptimizer=value)

    def setOutputLogsPath(self, p):
        return self._set(outputLogsPath=p)

    @keyword_only
    def __init__(self):
        super(NerDLApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach")
        uc = False if sys.platform == 'win32' else True
        self._setDefault(
            minEpochs=0,
            maxEpochs=50,
            lr=float(0.001),
            po=float(0.005),
            batchSize=8,
            dropout=float(0.5),
            verbose=2,
            useContrib=uc,
            validationSplit=float(0.0),
            evaluationLogExtended=False,
            includeConfidence=False,
            enableOutputLogs=False,
            enableMemoryOptimizer=False
        )


class NerDLModel(AnnotatorModel, HasStorageRef):
    name = "NerDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel", java_model=None):
        super(NerDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(includeConfidence=False)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)
    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "whether to include confidence scores in annotation metadata",
                              TypeConverters.toBoolean)
    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this NerDLModel",
                    TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setIncludeConfidence(self, value):
        return self._set(includeConfidence=value)

    @staticmethod
    def pretrained(name="ner_dl", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NerDLModel, name, lang, remote_loc)


class NerConverter(AnnotatorModel):
    name = 'NerConverter'

    whiteList = Param(
        Params._dummy(),
        "whiteList",
        "If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels",
        typeConverter=TypeConverters.toListString
    )

    def setWhiteList(self, entities):
        return self._set(whiteList=entities)

    @keyword_only
    def __init__(self):
        super(NerConverter, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.NerConverter")


class DependencyParserApproach(AnnotatorApproach):
    dependencyTreeBank = Param(Params._dummy(),
                               "dependencyTreeBank",
                               "Dependency treebank source files",
                               typeConverter=TypeConverters.identity)

    conllU = Param(Params._dummy(),
                   "conllU",
                   "Universal Dependencies source files",
                   typeConverter=TypeConverters.identity)

    numberOfIterations = Param(Params._dummy(),
                               "numberOfIterations",
                               "Number of iterations in training, converges to better accuracy",
                               typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(DependencyParserApproach,
              self).__init__(classname="com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach")
        self._setDefault(numberOfIterations=10)

    def setNumberOfIterations(self, value):
        return self._set(numberOfIterations=value)

    def setDependencyTreeBank(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        opts = options.copy()
        return self._set(dependencyTreeBank=ExternalResource(path, read_as, opts))

    def setConllU(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        opts = options.copy()
        return self._set(conllU=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return DependencyParserModel(java_model=java_model)


class DependencyParserModel(AnnotatorModel):
    name = "DependencyParserModel"

    perceptron = Param(Params._dummy(),
                       "perceptron",
                       "Dependency parsing perceptron features",
                       typeConverter=TypeConverters.identity)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel", java_model=None):
        super(DependencyParserModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="dependency_conllu", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(DependencyParserModel, name, lang, remote_loc)


class TypedDependencyParserApproach(AnnotatorApproach):
    conll2009 = Param(Params._dummy(),
                      "conll2009",
                      "Path to file with CoNLL 2009 format",
                      typeConverter=TypeConverters.identity)

    conllU = Param(Params._dummy(),
                   "conllU",
                   "Universal Dependencies source files",
                   typeConverter=TypeConverters.identity)

    numberOfIterations = Param(Params._dummy(),
                               "numberOfIterations",
                               "Number of iterations in training, converges to better accuracy",
                               typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(TypedDependencyParserApproach,
              self).__init__(classname="com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach")

    def setConll2009(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        opts = options.copy()
        return self._set(conll2009=ExternalResource(path, read_as, opts))

    def setConllU(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        opts = options.copy()
        return self._set(conllU=ExternalResource(path, read_as, opts))

    def setNumberOfIterations(self, value):
        return self._set(numberOfIterations=value)

    def _create_model(self, java_model):
        return TypedDependencyParserModel(java_model=java_model)


class TypedDependencyParserModel(AnnotatorModel):

    name = "TypedDependencyParserModel"

    trainOptions = Param(Params._dummy(),
                         "trainOptions",
                         "Training Options",
                         typeConverter=TypeConverters.identity)

    trainParameters = Param(Params._dummy(),
                            "trainParameters",
                            "Training Parameters",
                            typeConverter=TypeConverters.identity)

    trainDependencyPipe = Param(Params._dummy(),
                                "trainDependencyPipe",
                                "Training dependency pipe",
                                typeConverter=TypeConverters.identity)

    conllFormat = Param(Params._dummy(),
                        "conllFormat",
                        "CoNLL Format",
                        typeConverter=TypeConverters.toString)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel",
                 java_model=None):
        super(TypedDependencyParserModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="dependency_typed_conllu", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TypedDependencyParserModel, name, lang, remote_loc)


class WordEmbeddings(AnnotatorApproach, HasEmbeddingsProperties, HasStorage):

    name = "WordEmbeddings"

    writeBufferSize = Param(Params._dummy(),
                            "writeBufferSize",
                            "buffer size limit before dumping to disk storage while writing",
                            typeConverter=TypeConverters.toInt)

    readCacheSize = Param(Params._dummy(),
                          "readCacheSize",
                          "cache size for items retrieved from storage. Increase for performance but higher memory consumption",
                          typeConverter=TypeConverters.toInt)

    def setWriteBufferSize(self, v):
        return self._set(writeBufferSize=v)

    def setReadCacheSize(self, v):
        return self._set(readCacheSize=v)

    @keyword_only
    def __init__(self):
        super(WordEmbeddings, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.WordEmbeddings")
        self._setDefault(
            caseSensitive=False,
            writeBufferSize=10000,
            storageRef=self.uid
        )

    def _create_model(self, java_model):
        return WordEmbeddingsModel(java_model=java_model)


class WordEmbeddingsModel(AnnotatorModel, HasEmbeddingsProperties, HasStorageModel):

    name = "WordEmbeddingsModel"
    databases = ['EMBEDDINGS']

    readCacheSize = Param(Params._dummy(),
                          "readCacheSize",
                          "cache size for items retrieved from storage. Increase for performance but higher memory consumption",
                          typeConverter=TypeConverters.toInt)

    def setReadCacheSize(self, v):
        return self._set(readCacheSize=v)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel", java_model=None):
        super(WordEmbeddingsModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def overallCoverage(dataset, embeddings_col):
        from sparknlp.internal import _EmbeddingsOverallCoverage
        from sparknlp.common import CoverageResult
        return CoverageResult(_EmbeddingsOverallCoverage(dataset, embeddings_col).apply())

    @staticmethod
    def withCoverageColumn(dataset, embeddings_col, output_col='coverage'):
        from sparknlp.internal import _EmbeddingsCoverageColumn
        from pyspark.sql import DataFrame
        return DataFrame(_EmbeddingsCoverageColumn(dataset, embeddings_col, output_col).apply(), dataset.sql_ctx)

    @staticmethod
    def pretrained(name="glove_100d", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(WordEmbeddingsModel, name, lang, remote_loc)

    @staticmethod
    def loadStorage(path, spark, storage_ref):
        HasStorageModel.loadStorages(path, spark, storage_ref, WordEmbeddingsModel.databases)


class BertEmbeddings(AnnotatorModel, HasEmbeddingsProperties, HasCaseSensitiveProperties, HasStorageRef):

    name = "BertEmbeddings"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Batch size. Large values allows faster processing but requires more memory.",
                      typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        return self._set(maxSentenceLength=value)

    def setBatchSize(self, value):
        return self._set(batchSize=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.BertEmbeddings", java_model=None):
        super(BertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=32,
            maxSentenceLength=128,
            caseSensitive=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        from sparknlp.internal import _BertLoader
        jModel = _BertLoader(folder, spark_session._jsparkSession)._java_obj
        return BertEmbeddings(java_model=jModel)


    @staticmethod
    def pretrained(name="small_bert_L2_768", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BertEmbeddings, name, lang, remote_loc)


class BertSentenceEmbeddings(AnnotatorModel, HasEmbeddingsProperties, HasCaseSensitiveProperties, HasStorageRef):

    name = "BertSentenceEmbeddings"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Batch size. Large values allows faster processing but requires more memory.",
                      typeConverter=TypeConverters.toInt)

    isLong = Param(Params._dummy(),
                   "isLong",
                   "Use Long type instead of Int type for inputs buffer - Some Bert models require Long instead of Int.",
                   typeConverter=TypeConverters.toBoolean)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        return self._set(maxSentenceLength=value)

    def setBatchSize(self, value):
        return self._set(batchSize=value)

    def setIsLong(self, value):
        return self._set(isLong=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings", java_model=None):
        super(BertSentenceEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=32,
            maxSentenceLength=128,
            caseSensitive=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        from sparknlp.internal import _BertSentenceLoader
        jModel = _BertSentenceLoader(folder, spark_session._jsparkSession)._java_obj
        return BertSentenceEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="sent_small_bert_L2_768", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BertSentenceEmbeddings, name, lang, remote_loc)


class SentenceEmbeddings(AnnotatorModel, HasEmbeddingsProperties, HasStorageRef):

    name = "SentenceEmbeddings"

    @keyword_only
    def __init__(self):
        super(SentenceEmbeddings, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings")
        self._setDefault(
            poolingStrategy="AVERAGE"
        )

    poolingStrategy = Param(Params._dummy(),
                            "poolingStrategy",
                            "Choose how you would like to aggregate Word Embeddings to Sentence Embeddings: AVERAGE or SUM",
                            typeConverter=TypeConverters.toString)

    def setPoolingStrategy(self, strategy):
        if strategy == "AVERAGE":
            return self._set(poolingStrategy=strategy)
        elif strategy == "SUM":
            return self._set(poolingStrategy=strategy)
        else:
            return self._set(poolingStrategy="AVERAGE")


class StopWordsCleaner(AnnotatorModel):

    name = "StopWordsCleaner"

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.StopWordsCleaner", java_model=None):
        super(StopWordsCleaner, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            stopWords=StopWordsCleaner.loadDefaultStopWords("english"),
            caseSensitive=False,
            locale=self._java_obj.getLocale()
        )

    stopWords = Param(Params._dummy(), "stopWords", "The words to be filtered out",
                      typeConverter=TypeConverters.toListString)
    caseSensitive = Param(Params._dummy(), "caseSensitive", "whether to do a case sensitive " +
                          "comparison over the stop words", typeConverter=TypeConverters.toBoolean)
    locale = Param(Params._dummy(), "locale", "locale of the input. ignored when case sensitive " +
                   "is true", typeConverter=TypeConverters.toString)

    def setStopWords(self, value):
        return self._set(stopWords=value)

    def setCaseSensitive(self, value):
        return self._set(caseSensitive=value)

    def setLocale(self, value):
        return self._set(locale=value)

    def loadDefaultStopWords(language="english"):
        from pyspark.ml.wrapper import _jvm

        """
        Loads the default stop words for the given language.
        Supported languages: danish, dutch, english, finnish, french, german, hungarian,
        italian, norwegian, portuguese, russian, spanish, swedish, turkish
        """
        stopWordsObj = _jvm().org.apache.spark.ml.feature.StopWordsRemover
        return list(stopWordsObj.loadDefaultStopWords(language))

    @staticmethod
    def pretrained(name="stopwords_en", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(StopWordsCleaner, name, lang, remote_loc)


class NGramGenerator(AnnotatorModel):

    name = "NGramGenerator"

    @keyword_only
    def __init__(self):
        super(NGramGenerator, self).__init__(classname="com.johnsnowlabs.nlp.annotators.NGramGenerator")
        self._setDefault(
            n=2,
            enableCumulative=False
        )

    n = Param(Params._dummy(), "n", "number elements per n-gram (>=1)", typeConverter=TypeConverters.toInt)
    enableCumulative = Param(Params._dummy(), "enableCumulative", "whether to calculate just the actual n-grams " +
                             "or all n-grams from 1 through n", typeConverter=TypeConverters.toBoolean)

    delimiter = Param(Params._dummy(), "delimiter", "String to use to join the tokens ", typeConverter=TypeConverters.toString)

    def setN(self, value):
        """
        Sets the value of :py:attr:`n`.
        """
        return self._set(n=value)

    def setEnableCumulative(self, value):
        """
        Sets the value of :py:attr:`enableCumulative`.
        """
        return self._set(enableCumulative=value)

    def setDelimiter(self, value):
        """
        Sets the value of :py:attr:`delimiter`.
        """
        if len(value) > 1:
            raise Exception("Delimiter should have length == 1")
        return self._set(delimiter=value)


class ChunkEmbeddings(AnnotatorModel):

    name = "ChunkEmbeddings"

    @keyword_only
    def __init__(self):
        super(ChunkEmbeddings, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings")
        self._setDefault(
            poolingStrategy="AVERAGE"
        )

    poolingStrategy = Param(Params._dummy(),
                            "poolingStrategy",
                            "Choose how you would like to aggregate Word Embeddings to Chunk Embeddings:" +
                            "AVERAGE or SUM",
                            typeConverter=TypeConverters.toString)
    skipOOV = Param(Params._dummy(), "skipOOV", "Whether to discard default vectors for OOV words from the aggregation / pooling ", typeConverter=TypeConverters.toBoolean)

    def setPoolingStrategy(self, strategy):
        """
        Sets the value of :py:attr:`poolingStrategy`.
        """
        if strategy == "AVERAGE":
            return self._set(poolingStrategy=strategy)
        elif strategy == "SUM":
            return self._set(poolingStrategy=strategy)
        else:
            return self._set(poolingStrategy="AVERAGE")

    def setSkipOOV(self, value):
        """
        Sets the value of :py:attr:`skipOOV`.
        """
        return self._set(skipOOV=value)


class NerOverwriter(AnnotatorModel):

    name = "NerOverwriter"

    @keyword_only
    def __init__(self):
        super(NerOverwriter, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.NerOverwriter")
        self._setDefault(
            newResult="I-OVERWRITE"
        )

    stopWords = Param(Params._dummy(), "stopWords", "The words to be overwritten",
                      typeConverter=TypeConverters.toListString)
    newResult = Param(Params._dummy(), "newResult", "new NER class to apply to those stopwords",
                      typeConverter=TypeConverters.toString)

    def setStopWords(self, value):
        return self._set(stopWords=value)

    def setNewResult(self, value):
        return self._set(newResult=value)


class UniversalSentenceEncoder(AnnotatorModel, HasEmbeddingsProperties, HasStorageRef):

    name = "UniversalSentenceEncoder"

    loadSP = Param(Params._dummy(), "loadSP", "Whether to load SentencePiece ops file which is required only by multi-lingual models. "
                                              "This is not changeable after it's set with a pretrained model nor it is compatible with Windows.", typeConverter=TypeConverters.toBoolean)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setLoadSP(self, value):
        """
        Sets the value of :py:attr:`loadSP`.
        """
        return self._set(loadSP=value)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder", java_model=None):
        super(UniversalSentenceEncoder, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            loadSP=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session, loadsp=False):
        from sparknlp.internal import _USELoader
        jModel = _USELoader(folder, spark_session._jsparkSession, loadsp)._java_obj
        return UniversalSentenceEncoder(java_model=jModel)


    @staticmethod
    def pretrained(name="tfhub_use", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(UniversalSentenceEncoder, name, lang, remote_loc)


class ElmoEmbeddings(AnnotatorModel, HasEmbeddingsProperties, HasCaseSensitiveProperties, HasStorageRef):

    name = "ElmoEmbeddings"

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Batch size. Large values allows faster processing but requires more memory.",
                      typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    poolingLayer = Param(Params._dummy(),
                         "poolingLayer", "Set ELMO pooling layer to: word_emb, lstm_outputs1, lstm_outputs2, or elmo",
                         typeConverter=TypeConverters.toString)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setBatchSize(self, value):
        return self._set(batchSize=value)

    def setPoolingLayer(self, layer):
        if layer == "word_emb":
            return self._set(poolingLayer=layer)
        elif layer == "lstm_outputs1":
            return self._set(poolingLayer=layer)
        elif layer == "lstm_outputs2":
            return self._set(poolingLayer=layer)
        elif layer == "elmo":
            return self._set(poolingLayer=layer)
        else:
            return self._set(poolingLayer="word_emb")

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings", java_model=None):
        super(ElmoEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=32,
            poolingLayer="word_emb"
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        from sparknlp.internal import _ElmoLoader
        jModel = _ElmoLoader(folder, spark_session._jsparkSession)._java_obj
        return ElmoEmbeddings(java_model=jModel)


    @staticmethod
    def pretrained(name="elmo", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ElmoEmbeddings, name, lang, remote_loc)


class ClassifierDLApproach(AnnotatorApproach):

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)

    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)

    validationSplit = Param(Params._dummy(), "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

    outputLogsPath = Param(Params._dummy(), "outputLogsPath", "Folder path to save training logs", TypeConverters.toString)

    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Column with label per each token",
                        typeConverter=TypeConverters.toString)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)
    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)

    def setVerbose(self, value):
        return self._set(verbose=value)

    def setRandomSeed(self, seed):
        return self._set(randomSeed=seed)

    def setLabelColumn(self, value):
        return self._set(labelColumn=value)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setLr(self, v):
        self._set(lr=v)
        return self

    def setBatchSize(self, v):
        self._set(batchSize=v)
        return self

    def setDropout(self, v):
        self._set(dropout=v)
        return self

    def setMaxEpochs(self, epochs):
        return self._set(maxEpochs=epochs)

    def _create_model(self, java_model):
        return ClassifierDLModel(java_model=java_model)

    def setValidationSplit(self, v):
        self._set(validationSplit=v)
        return self

    def setEnableOutputLogs(self, value):
        return self._set(enableOutputLogs=value)

    def setOutputLogsPath(self, p):
        return self._set(outputLogsPath=p)

    @keyword_only
    def __init__(self):
        super(ClassifierDLApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach")
        self._setDefault(
            maxEpochs=30,
            lr=float(0.005),
            batchSize=64,
            dropout=float(0.5),
            enableOutputLogs=False
        )


class ClassifierDLModel(AnnotatorModel, HasStorageRef):
    name = "ClassifierDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel", java_model=None):
        super(ClassifierDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)

    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this ClassifierDLModel",
                    TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    @staticmethod
    def pretrained(name="classifierdl_use_trec6", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ClassifierDLModel, name, lang, remote_loc)


class AlbertEmbeddings(AnnotatorModel, HasEmbeddingsProperties, HasCaseSensitiveProperties, HasStorageRef):

    name = "AlbertEmbeddings"

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Batch size. Large values allows faster processing but requires more memory.",
                      typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setBatchSize(self, value):
        return self._set(batchSize=value)

    def setMaxSentenceLength(self, value):
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings", java_model=None):
        super(AlbertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=32,
            dimension=768,
            maxSentenceLength=128
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        from sparknlp.internal import _AlbertLoader
        jModel = _AlbertLoader(folder, spark_session._jsparkSession)._java_obj
        return AlbertEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="albert_base_uncased", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(AlbertEmbeddings, name, lang, remote_loc)


class XlnetEmbeddings(AnnotatorModel, HasEmbeddingsProperties, HasCaseSensitiveProperties, HasStorageRef):

    name = "XlnetEmbeddings"

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Batch size. Large values allows faster processing but requires more memory.",
                      typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setBatchSize(self, value):
        return self._set(batchSize=value)

    def setMaxSentenceLength(self, value):
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.XlnetEmbeddings", java_model=None):
        super(XlnetEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=32,
            dimension=768,
            maxSentenceLength=128
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        from sparknlp.internal import _XlnetLoader
        jModel = _XlnetLoader(folder, spark_session._jsparkSession)._java_obj
        return XlnetEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="xlnet_base_cased", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(XlnetEmbeddings, name, lang, remote_loc)


class ContextSpellCheckerApproach(AnnotatorApproach):

    name = "ContextSpellCheckerApproach"

    languageModelClasses = Param(Params._dummy(),
                                 "languageModelClasses",
                                 "Number of classes to use during factorization of the softmax output in the LM.",
                                 typeConverter=TypeConverters.toInt)

    wordMaxDistance = Param(Params._dummy(),
                            "wordMaxDistance",
                            "Maximum distance for the generated candidates for every word.",
                            typeConverter=TypeConverters.toInt)

    maxCandidates = Param(Params._dummy(),
                          "maxCandidates",
                          "Maximum number of candidates for every word.",
                          typeConverter=TypeConverters.toInt)

    caseStrategy = Param(Params._dummy(),
                         "caseStrategy",
                         "What case combinations to try when generating candidates.",
                         typeConverter=TypeConverters.toInt)

    errorThreshold = Param(Params._dummy(),
                           "errorThreshold",
                           "Threshold perplexity for a word to be considered as an error.",
                           typeConverter=TypeConverters.toFloat)

    epochs = Param(Params._dummy(),
                   "epochs",
                   "Number of epochs to train the language model.",
                   typeConverter=TypeConverters.toInt)

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Batch size for the training in NLM.",
                      typeConverter=TypeConverters.toInt)

    initialRate = Param(Params._dummy(),
                        "initialRate",
                        "Initial learning rate for the LM.",
                        typeConverter=TypeConverters.toFloat)

    finalRate = Param(Params._dummy(),
                      "finalRate",
                      "Final learning rate for the LM.",
                      typeConverter=TypeConverters.toFloat)

    validationFraction = Param(Params._dummy(),
                               "validationFraction",
                               "Percentage of datapoints to use for validation.",
                               typeConverter=TypeConverters.toFloat)

    minCount = Param(Params._dummy(),
                     "minCount",
                     "Min number of times a token should appear to be included in vocab.",
                     typeConverter=TypeConverters.toInt)

    compoundCount = Param(Params._dummy(),
                          "compoundCount",
                          "Min number of times a compound word should appear to be included in vocab.",
                          typeConverter=TypeConverters.toInt)

    classCount = Param(Params._dummy(),
                       "classCount",
                       "Min number of times the word need to appear in corpus to not be considered of a special class.",
                       typeConverter=TypeConverters.toInt)

    tradeoff = Param(Params._dummy(),
                     "tradeoff",
                     "Tradeoff between the cost of a word error and a transition in the language model.",
                     typeConverter=TypeConverters.toFloat)

    weightedDistPath = Param(Params._dummy(),
                             "weightedDistPath",
                             "The path to the file containing the weights for the levenshtein distance.",
                             typeConverter=TypeConverters.toString)

    maxWindowLen = Param(Params._dummy(),
                         "maxWindowLen",
                         "Maximum size for the window used to remember history prior to every correction.",
                         typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)


    def setLanguageModelClasses(self, count):
        return self._set(languageModelClasses=count)

    def setWordMaxDistance(self, dist):
        return self._set(wordMaxDistance=dist)

    def setMaxCandidates(self, candidates):
        return self._set(maxCandidates=candidates)

    def setCaseStrategy(self, strategy):
        return self._set(caseStrategy=strategy)

    def setErrorThreshold(self, threshold):
        return self._set(errorThreshold=threshold)

    def setEpochs(self, count):
        return self._set(epochs=count)

    def setInitialBatchSize(self, size):
        return self._set(batchSize=size)

    def setInitialRate(self, rate):
        return self._set(initialRate=rate)

    def setFinalRate(self, rate):
        return self._set(finalRate=rate)

    def setValidationFraction(self, fraction):
        return self._set(validationFraction=fraction)

    def setMinCount(self, count):
        return self._set(minCount=count)

    def setCompoundCount(self, count):
        return self._set(compoundCount=count)

    def setClassCount(self, count):
        return self._set(classCount=count)

    def setTradeoff(self, alpha):
        return self._set(tradeoff=alpha)

    def setWeightedDistPath(self, path):
        return self._set(weightedDistPath=path)

    def setWeightedDistPath(self, path):
        return self._set(weightedDistPath=path)

    def setMaxWindowLen(self, length):
        return self._set(maxWindowLen=length)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def addVocabClass(self, label, vocab, userdist=3):
        self._call_java('addVocabClass', label, vocab, userdist)
        return self

    def addRegexClass(self, label, regex, userdist=3):
        self._call_java('addRegexClass', label, regex, userdist)
        return self

    @keyword_only
    def __init__(self):
        super(ContextSpellCheckerApproach, self). \
            __init__(classname="com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerApproach")

    def _create_model(self, java_model):
        return ContextSpellCheckerModel(java_model=java_model)


class ContextSpellCheckerModel(AnnotatorModel):
    name = "ContextSpellCheckerModel"

    wordMaxDistance = Param(Params._dummy(),
                            "wordMaxDistance",
                            "Maximum distance for the generated candidates for every word.",
                            typeConverter=TypeConverters.toInt)

    maxCandidates = Param(Params._dummy(),
                          "maxCandidates",
                          "Maximum number of candidates for every word.",
                          typeConverter=TypeConverters.toInt)

    caseStrategy = Param(Params._dummy(),
                         "caseStrategy",
                         "What case combinations to try when generating candidates.",
                         typeConverter=TypeConverters.toInt)

    errorThreshold = Param(Params._dummy(),
                           "errorThreshold",
                           "Threshold perplexity for a word to be considered as an error.",
                           typeConverter=TypeConverters.toFloat)

    tradeoff = Param(Params._dummy(),
                     "tradeoff",
                     "Tradeoff between the cost of a word error and a transition in the language model.",
                     typeConverter=TypeConverters.toFloat)

    weightedDistPath = Param(Params._dummy(),
                             "weightedDistPath",
                             "The path to the file containing the weights for the levenshtein distance.",
                             typeConverter=TypeConverters.toString)

    maxWindowLen = Param(Params._dummy(),
                         "maxWindowLen",
                         "Maximum size for the window used to remember history prior to every correction.",
                         typeConverter=TypeConverters.toInt)

    gamma = Param(Params._dummy(),
                  "gamma",
                  "Controls the influence of individual word frequency in the decision.",
                  typeConverter=TypeConverters.toFloat)

    correctSymbols = Param(Params._dummy(), "correctSymbols", "Whether to correct special symbols or skip spell checking for them", typeConverter=TypeConverters.toBoolean)

    compareLowcase = Param(Params._dummy(), "compareLowcase", "If true will compare tokens in low case with vocabulary", typeConverter=TypeConverters.toBoolean)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)


    def setWordMaxDistance(self, dist):
        return self._set(wordMaxDistance=dist)

    def setMaxCandidates(self, candidates):
        return self._set(maxCandidates=candidates)

    def setCaseStrategy(self, strategy):
        return self._set(caseStrategy=strategy)

    def setErrorThreshold(self, threshold):
        return self._set(errorThreshold=threshold)

    def setTradeoff(self, alpha):
        return self._set(tradeoff=alpha)

    def setWeights(self, weights):
        self._call_java('setWeights', weights)

    def setMaxWindowLen(self, length):
        return self._set(maxWindowLen=length)

    def setGamma(self, g):
        return self._set(gamma=g)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def getWordClasses(self):
        it = self._call_java('getWordClasses').toIterator()
        result = []
        while(it.hasNext()):
            result.append(it.next().toString())
        return result

    def updateRegexClass(self, label, regex):
        self._call_java('updateRegexClass', label, regex)
        return self

    def updateVocabClass(self, label, vocab, append=True):
        self._call_java('updateVocabClass', label, vocab, append)
        return self

    def setCorrectSymbols(self, value):
        return self._set(correctSymbols=value)

    def setCompareLowcase(self, value):
        return self._set(compareLowcase=value)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel", java_model=None):
        super(ContextSpellCheckerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="spellcheck_dl", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ContextSpellCheckerModel, name, lang, remote_loc)


class SentimentDLApproach(AnnotatorApproach):

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)

    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)

    validationSplit = Param(Params._dummy(), "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

    outputLogsPath = Param(Params._dummy(), "outputLogsPath", "Folder path to save training logs", TypeConverters.toString)

    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Column with label per each token",
                        typeConverter=TypeConverters.toString)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)
    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)
    threshold = Param(Params._dummy(), "threshold", "The minimum threshold for the final result otheriwse it will be neutral", TypeConverters.toFloat)
    thresholdLabel = Param(Params._dummy(), "thresholdLabel", "In case the score is less than threshold, what should be the label. Default is neutral.", TypeConverters.toString)

    def setVerbose(self, value):
        return self._set(verbose=value)

    def setRandomSeed(self, seed):
        return self._set(randomSeed=seed)

    def setLabelColumn(self, value):
        return self._set(labelColumn=value)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setLr(self, v):
        self._set(lr=v)
        return self

    def setBatchSize(self, v):
        self._set(batchSize=v)
        return self

    def setDropout(self, v):
        self._set(dropout=v)
        return self

    def setMaxEpochs(self, epochs):
        return self._set(maxEpochs=epochs)

    def _create_model(self, java_model):
        return SentimentDLModel(java_model=java_model)

    def setValidationSplit(self, v):
        self._set(validationSplit=v)
        return self

    def setEnableOutputLogs(self, value):
        return self._set(enableOutputLogs=value)

    def setOutputLogsPath(self, p):
        return self._set(outputLogsPath=p)

    def setThreshold(self, v):
        self._set(threshold=v)
        return self

    def setThresholdLabel(self, p):
        return self._set(thresholdLabel=p)

    @keyword_only
    def __init__(self):
        super(SentimentDLApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLApproach")
        self._setDefault(
            maxEpochs=30,
            lr=float(0.005),
            batchSize=64,
            dropout=float(0.5),
            enableOutputLogs=False,
            threshold=0.6,
            thresholdLabel="neutral"
        )


class SentimentDLModel(AnnotatorModel, HasStorageRef):
    name = "SentimentDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel", java_model=None):
        super(SentimentDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            threshold=0.6,
            thresholdLabel="neutral"
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)
    threshold = Param(Params._dummy(), "threshold", "The minimum threshold for the final result otheriwse it will be neutral", TypeConverters.toFloat)
    thresholdLabel = Param(Params._dummy(), "thresholdLabel", "In case the score is less than threshold, what should be the label. Default is neutral.", TypeConverters.toString)
    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this SentimentDLModel",
                    TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setThreshold(self, v):
        self._set(threshold=v)
        return self

    def setThresholdLabel(self, p):
        return self._set(thresholdLabel=p)

    @staticmethod
    def pretrained(name="sentimentdl_use_imdb", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SentimentDLModel, name, lang, remote_loc)


class LanguageDetectorDL(AnnotatorModel, HasStorageRef):
    name = "LanguageDetectorDL"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL", java_model=None):
        super(LanguageDetectorDL, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            threshold=0.5,
            thresholdLabel="Unknown",
            coalesceSentences=True
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)
    threshold = Param(Params._dummy(), "threshold", "The minimum threshold for the final result otheriwse it will be either neutral or the value set in thresholdLabel.", TypeConverters.toFloat)
    thresholdLabel = Param(Params._dummy(), "thresholdLabel", "In case the score is less than threshold, what should be the label. Default is neutral.", TypeConverters.toString)
    coalesceSentences = Param(Params._dummy(), "coalesceSentences", "If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to false.", TypeConverters.toBoolean)
    languages = Param(Params._dummy(), "languages",
                      "get the languages used to trained the model",
                      TypeConverters.toListString)
    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setThreshold(self, v):
        self._set(threshold=v)
        return self

    def setThresholdLabel(self, p):
        return self._set(thresholdLabel=p)

    def setCoalesceSentences(self, value):
        return self._set(coalesceSentences=value)

    @staticmethod
    def pretrained(name="ld_wiki_tatoeba_cnn_21", lang="xx", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(LanguageDetectorDL, name, lang, remote_loc)


class MultiClassifierDLApproach(AnnotatorApproach):

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)

    validationSplit = Param(Params._dummy(), "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

    outputLogsPath = Param(Params._dummy(), "outputLogsPath", "Folder path to save training logs", TypeConverters.toString)

    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Column with label per each token",
                        typeConverter=TypeConverters.toString)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)
    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)
    shufflePerEpoch = Param(Params._dummy(), "shufflePerEpoch", "whether to shuffle the training data on each Epoch", TypeConverters.toBoolean)
    threshold = Param(Params._dummy(), "threshold", "The minimum threshold for each label to be accepted. Default is 0.5", TypeConverters.toFloat)

    def setVerbose(self, v):
        return self._set(verbose=v)

    def setRandomSeed(self, seed):
        return self._set(randomSeed=seed)

    def setLabelColumn(self, v):
        return self._set(labelColumn=v)

    def setConfigProtoBytes(self, v):
        return self._set(configProtoBytes=v)

    def setLr(self, v):
        self._set(lr=v)
        return self

    def setBatchSize(self, v):
        self._set(batchSize=v)
        return self

    def setMaxEpochs(self, v):
        return self._set(maxEpochs=v)

    def _create_model(self, java_model):
        return ClassifierDLModel(java_model=java_model)

    def setValidationSplit(self, v):
        self._set(validationSplit=v)
        return self

    def setEnableOutputLogs(self, v):
        return self._set(enableOutputLogs=v)

    def setOutputLogsPath(self, v):
        return self._set(outputLogsPath=v)

    def setShufflePerEpoch(self, v):
        return self._set(shufflePerEpoch=v)

    def setThreshold(self, v):
        self._set(threshold=v)
        return self

    @keyword_only
    def __init__(self):
        super(MultiClassifierDLApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach")
        self._setDefault(
            maxEpochs=10,
            lr=float(0.001),
            batchSize=64,
            validationSplit=float(0.0),
            threshold=float(0.5),
            randomSeed=44,
            shufflePerEpoch=False,
            enableOutputLogs=False
        )


class MultiClassifierDLModel(AnnotatorModel, HasStorageRef):
    name = "MultiClassifierDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLModel", java_model=None):
        super(MultiClassifierDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            threshold=float(0.5)
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)
    threshold = Param(Params._dummy(), "threshold", "The minimum threshold for each label to be accepted. Default is 0.5", TypeConverters.toFloat)
    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this MultiClassifierDLModel",
                    TypeConverters.toListString)

    def setThreshold(self, v):
        self._set(threshold=v)
        return self

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    @staticmethod
    def pretrained(name="multiclassifierdl_use_toxic", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(MultiClassifierDLModel, name, lang, remote_loc)


class YakeModel(AnnotatorModel):
    name = "YakeModel"
    @keyword_only
    def __init__(self):
        super(YakeModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.keyword.yake.YakeModel")
        self._setDefault(
            minNGrams=2,
            maxNGrams=3,
            nKeywords=30,
            windowSize=3,
            threshold=-1,
            stopWords=YakeModel.loadDefaultStopWords("english")
        )

    minNGrams = Param(Params._dummy(), "minNGrams", "Minimum N-grams a keyword should have", typeConverter=TypeConverters.toInt)
    maxNGrams = Param(Params._dummy(), "maxNGrams", "Maximum N-grams a keyword should have", typeConverter=TypeConverters.toInt)
    threshold = Param(Params._dummy(), "threshold", "Keyword Score threshold", typeConverter=TypeConverters.toFloat)
    windowSize = Param(Params._dummy(), "windowSize", "Window size for Co-Occurrence", typeConverter=TypeConverters.toInt)
    nKeywords = Param(Params._dummy(), "nKeywords", "Number of Keywords to extract", typeConverter=TypeConverters.toInt)
    stopWords = Param(Params._dummy(), "stopWords", "the words to be filtered out. by default it's english stop words from Spark ML",typeConverter=TypeConverters.toListString)

    def setWindowSize(self, value):
        return self._set(windowSize=value)

    def setMinNGrams(self, value):
        return self._set(minNGrams=value)

    def setMaxNGrams(self, value):
        return self._set(maxNGrams=value)

    def setThreshold(self, value):
        return self._set(threshold=value)

    def setNKeywords(self, value):
        return self._set(nKeywords=value)

    def setStopWords(self, value):
        return self._set(stopWords=value)

    def getStopWords(self):
        return self.getOrDefault(self.stopWords)

    def loadDefaultStopWords(language="english"):
        from pyspark.ml.wrapper import _jvm

        """
        Loads the default stop words for the given language.
        Supported languages: danish, dutch, english, finnish, french, german, hungarian,
        italian, norwegian, portuguese, russian, spanish, swedish, turkish
        """
        stopWordsObj = _jvm().org.apache.spark.ml.feature.StopWordsRemover
        return list(stopWordsObj.loadDefaultStopWords(language))


class SentenceDetectorDLModel(AnnotatorModel):
    name = "SentenceDetectorDLModel"

    modelArchitecture = Param(Params._dummy(), "modelArchitecture", "Model architecture (CNN)",
                              typeConverter=TypeConverters.toString)

    explodeSentences = Param(Params._dummy(),
                             "explodeSentences",
                             "whether to explode each sentence into a different row, for better parallelization. Defaults to false.",
                             TypeConverters.toBoolean)

    def setModel(self, modelArchitecture):
        return self._set(modelArchitecture=modelArchitecture)

    def setExplodeSentences(self, value):
        return self._set(explodeSentences=value)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel",
                 java_model=None):
        super(SentenceDetectorDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="sentence_detector_dl", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SentenceDetectorDLModel, name, lang, remote_loc)


class SentenceDetectorDLApproach(AnnotatorApproach):

    name = "SentenceDetectorDLApproach"

    modelArchitecture = Param(Params._dummy(),
                              "modelArchitecture",
                              "Model architecture (CNN)",
                              typeConverter=TypeConverters.toString)

    impossiblePenultimates = Param(Params._dummy(),
                                   "impossiblePenultimates",
                                   "Impossible penultimates - list of strings which a sentence can't end with",
                                   typeConverter=TypeConverters.toListString)

    validationSplit = Param(Params._dummy(),
                            "validationSplit",
                            "Choose the proportion of training dataset to be validated against the model on each "
                            "Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    epochsNumber = Param(Params._dummy(),
                         "epochsNumber",
                         "Number of epochs for the optimization process",
                         TypeConverters.toInt)

    outputLogsPath = Param(Params._dummy(),
                           "outputLogsPath",
                           "Path to folder where logs will be saved. If no path is specified, no logs are generated",
                           TypeConverters.toString)

    explodeSentences = Param(Params._dummy(),
                             "explodeSentences",
                             "whether to explode each sentence into a different row, for better parallelization. Defaults to false.",
                             TypeConverters.toBoolean)

    def setModel(self, model_architecture):
        return self._set(modelArchitecture=model_architecture)

    def setValidationSplit(self, validation_split):
        return self._set(validationSplit=validation_split)

    def setEpochsNumber(self, epochs_number):
        return self._set(epochsNumber=epochs_number)

    def setOutputLogsPath(self, output_logs_path):
        return self._set(outputLogsPath=output_logs_path)

    def setImpossiblePenultimates(self, impossible_penultimates):
        return self._set(impossiblePenultimates=impossible_penultimates)

    def setExplodeSentences(self, value):
        return self._set(explodeSentences=value)

    def _create_model(self, java_model):
        return SentenceDetectorDLModel(java_model=java_model)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach"):
        super(SentenceDetectorDLApproach, self).__init__(classname=classname)


class WordSegmenterApproach(AnnotatorApproach):
    name = "WordSegmenterApproach"

    posCol = Param(Params._dummy(),
                   "posCol",
                   "column of Array of POS tags that match tokens",
                   typeConverter=TypeConverters.toString)

    nIterations = Param(Params._dummy(),
                        "nIterations",
                        "Number of iterations in training, converges to better accuracy",
                        typeConverter=TypeConverters.toInt)

    frequencyThreshold = Param(Params._dummy(),
                               "frequencyThreshold",
                               "How many times at least a tag on a word to be marked as frequent",
                               typeConverter=TypeConverters.toInt)

    ambiguityThreshold = Param(Params._dummy(),
                               "ambiguityThreshold",
                               "How much percentage of total amount of words are covered to be marked as frequent",
                               typeConverter=TypeConverters.toFloat)

    @keyword_only
    def __init__(self):
        super(WordSegmenterApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.ws.WordSegmenterApproach")
        self._setDefault(
            nIterations=5, frequencyThreshold=5, ambiguityThreshold=0.97
        )

    def setPosCol(self, value):
        return self._set(posCol=value)

    def setIterations(self, value):
        return self._set(nIterations=value)

    def setFrequencyThreshold(self, value):
        return self._set(frequencyThreshold=value)

    def setAmbiguityThreshold(self, value):
        return self._set(ambiguityThreshold=value)

    def getNIterations(self):
        return self.getOrDefault(self.nIterations)
    
    def getFrequencyThreshold(self):
        return self.getOrDefault(self.frequencyThreshold)
    
    def getAmbiguityThreshold(self):
        return self.getOrDefault(self.ambiguityThreshold)

    def _create_model(self, java_model):
        return WordSegmenterModel(java_model=java_model)


class WordSegmenterModel(AnnotatorModel):
    name = "WordSegmenterModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ws.WordSegmenterModel", java_model=None):
        super(WordSegmenterModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="wordseg_pku", lang="zh", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(WordSegmenterModel, name, lang, remote_loc)


class T5Transformer(AnnotatorModel):

    name = "T5Transformer"

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    task = Param(Params._dummy(), "task", "Transformer's task, e.g. summarize>", typeConverter=TypeConverters.toString)

    maxOutputLength = Param(Params._dummy(), "maxOutputLength", "Set the maximum length of output text", typeConverter=TypeConverters.toInt)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setTask(self, value):
        return self._set(task=value)

    def setMaxOutputLength(self, value):
        return self._set(maxOutputLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer", java_model=None):
        super(T5Transformer, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        from sparknlp.internal import _T5Loader
        jModel = _T5Loader(folder, spark_session._jsparkSession)._java_obj
        return T5Transformer(java_model=jModel)

    @staticmethod
    def pretrained(name="t5_small", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(T5Transformer, name, lang, remote_loc)


class MarianTransformer(AnnotatorModel):

    name = "MarianTransformer"

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    langId = Param(Params._dummy(), "langId", "Transformer's task, e.g. summarize>", typeConverter=TypeConverters.toString)

    maxInputLength = Param(Params._dummy(), "maxInputLength", "Controls the maximum length for encoder inputs (source language texts)", typeConverter=TypeConverters.toInt)

    maxOutputLength = Param(Params._dummy(), "maxOutputLength", "Controls the maximum length for decoder outputs (target language texts)", typeConverter=TypeConverters.toInt)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setLangId(self, value):
        return self._set(langId=value)

    def setMaxInputLength(self, value):
        return self._set(maxInputLength=value)

    def setMaxOutputLength(self, value):
        return self._set(maxOutputLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer", java_model=None):
        super(MarianTransformer, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        from sparknlp.internal import _MarianLoader
        jModel = _MarianLoader(folder, spark_session._jsparkSession)._java_obj
        return MarianTransformer(java_model=jModel)

    @staticmethod
    def pretrained(name="opus_mt_en_fr", lang="xx", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(MarianTransformer, name, lang, remote_loc)
