import unittest
import re
from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest
from sparknlp.ocr import OcrHelper


class BasicAnnotatorsTestSpec(unittest.TestCase):

    def setUp(self):
        # This implicitly sets up py4j for us
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setOutputCol("token") \
            .setCompositeTokensPatterns(["New York"]) \
            .addInfixPattern("(%\\d+)") \
            .setIncludeDefaults(True)
        stemmer = Stemmer() \
            .setInputCols(["token"]) \
            .setOutputCol("stem")
        normalizer = Normalizer() \
            .setInputCols(["stem"]) \
            .setOutputCol("normalize")
        token_assembler = TokenAssembler() \
            .setInputCols(["normalize"]) \
            .setOutputCol("assembled")
        finisher = Finisher() \
            .setInputCols(["assembled"]) \
            .setOutputCols(["reassembled_view"]) \
            .setCleanAnnotations(True)
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.transform(assembled)
        stemmed = stemmer.transform(tokenized)
        normalized = normalizer.fit(stemmed).transform(stemmed)
        reassembled = token_assembler.transform(normalized)
        finisher.transform(reassembled).show()


class RegexMatcherTestSpec(unittest.TestCase):

    def setUp(self):
        # This implicitly sets up py4j for us
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        regex_matcher = RegexMatcher() \
            .setStrategy("MATCH_ALL") \
            .setExternalRules(path="file:///" + os.getcwd() + "/../src/test/resources/regex-matcher/rules.txt",
                              delimiter=",") \
            .setOutputCol("regex")
        assembled = document_assembler.transform(self.data)
        regex_matcher.fit(assembled).transform(assembled).show()


class LemmatizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma") \
            .setDictionary(path="file:///" + os.getcwd() + "/../src/test/resources/lemma-corpus-small/lemmas_small.txt",
                           key_delimiter="->", value_delimiter="\t")
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.transform(assembled)
        lemmatizer.fit(tokenized).transform(tokenized).show()


class TokenizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark

    def runTest(self):
        data = self.session.createDataFrame([("this is some/text I wrote",)], ["text"])
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setOutputCol("token") \
            .addInfixPattern("(\\p{L}+)\\/(\\p{L}+\\b)")
        finisher = Finisher() \
            .setInputCols(["token"]) \
            .setOutputCols(["token_out"]) \
            .setOutputAsArray(True)
        assembled = document_assembler.transform(data)
        tokenized = tokenizer.transform(assembled)
        finished = finisher.transform(tokenized)
        self.assertEqual(len(finished.first()['token_out']), 6)


class NormalizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setOutputCol("token")
        lemmatizer = Normalizer() \
            .setInputCols(["token"]) \
            .setOutputCol("normalized_token") \
            .setLowercase(False)

        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.transform(assembled)
        lemmatizer.transform(tokenized).show()


class DateMatcherTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        date_matcher = DateMatcher() \
            .setOutputCol("date") \
            .setDateFormat("yyyyMM")
        assembled = document_assembler.transform(self.data)
        date_matcher.transform(assembled).show()


class TextMatcherTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setOutputCol("token")
        entity_extractor = TextMatcher() \
            .setOutputCol("entity") \
            .setEntities(path="file:///" + os.getcwd() + "/../src/test/resources/entity-extractor/test-phrases.txt")
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.transform(assembled)
        entity_extractor.fit(tokenized).transform(tokenized).show()


class PerceptronApproachTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")
        pos_tagger = PerceptronApproach() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos") \
            .setCorpus(path="file:///" + os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/", delimiter="|") \
            .setIterations(2) \
            .fit(self.data)
        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.transform(sentenced)
        pos_tagger.transform(tokenized).show()


class ChunkerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")
        pos_tagger = PerceptronApproach() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos") \
            .setCorpus(path="file:///" + os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/", delimiter="|") \
            .setIterations(2) \
            .fit(self.data)
        chunker = Chunker() \
            .setInputCols(["pos"]) \
            .setOutputCol("chunk") \
            .setRegexParsers(["<NNP>+", "<DT|PP\\$>?<JJ>*<NN>"])
        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.transform(sentenced)
        pos_sentence_format = pos_tagger.transform(tokenized)
        chunk_phrases = chunker.transform(pos_sentence_format)
        chunk_phrases.show()


class DeIdentificationTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")

        ner_tagger = NerCrfModel.pretrained()

        ner_converter = NerConverter() \
            .setInputCols(["sentence", "token", "ner"]) \
            .setOutputCol("ner_con")

        de_identification = DeIdentification() \
            .setInputCols(["ner_con", "token", "document"]) \
            .setOutputCol("dei") \
            .setRegexPatternsDictionary("src/test/resources/de-identification/DicRegexPatterns.txt")

        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.transform(sentenced)
        ner_tagged = ner_tagger.transform(tokenized)
        ner_converted = ner_converter.transform(ner_tagged)
        de_identified = de_identification.fit(ner_converted).transform(ner_converted)
        de_identified.show()


class DependencyParserTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")

        pos_tagger = PerceptronApproach() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos") \
            .setCorpus(path="file:///" + os.getcwd() +
                            "/../../src/test/resources/anc-pos-corpus-small/", delimiter="|") \
            .setIterations(1)

        dependency_parser = DependencyParserApproach() \
            .setInputCols(["sentence", "pos", "token"]) \
            .setOutputCol("dependency") \
            .setDependencyTreeBank(path=os.getcwd() + "/../../src/test/resources/parser/dependency_treebank") \
            .setNumberOfIterations(10)

        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.transform(sentenced)
        pos_tagged = pos_tagger.fit(tokenized).transform(tokenized)
        dependency_parsed = dependency_parser.fit(pos_tagged).transform(pos_tagged)
        dependency_parsed.show()


class PragmaticSBDTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence") \
            .setCustomBounds(["%%"])
        assembled = document_assembler.transform(self.data)
        sentence_detector.transform(assembled).show()


class PragmaticScorerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma") \
            .setDictionary(path="file:///" + os.getcwd() + "/../src/test/resources/lemma-corpus-small/lemmas_small.txt",
                           key_delimiter="->", value_delimiter="\t")
        sentiment_detector = SentimentDetector() \
            .setInputCols(["lemma", "sentence"]) \
            .setOutputCol("sentiment") \
            .setDictionary(
            "file:///" + os.getcwd() + "/../src/test/resources/sentiment-corpus/default-sentiment-dict.txt",
            delimiter=",")
        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.transform(sentenced)
        lemmatized = lemmatizer.fit(tokenized).transform(tokenized)
        sentiment_detector.fit(lemmatized).transform(lemmatized).show()


class PipelineTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma") \
            .setDictionary("file:///" + os.getcwd() + "/../src/test/resources/lemma-corpus-small/simple.txt",
                           key_delimiter="->", value_delimiter="\t")
        finisher = Finisher() \
            .setInputCols(["token", "lemma"]) \
            .setOutputCols(["token_views", "lemma_views"])
        pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer, finisher])
        model = pipeline.fit(self.data)
        token_before_save = model.transform(self.data).select("token_views").take(1)[0].token_views.split("@")[2]
        lemma_before_save = model.transform(self.data).select("lemma_views").take(1)[0].lemma_views.split("@")[2]
        pipe_path = "file:///" + os.getcwd() + "/tmp_pipeline"
        pipeline.write().overwrite().save(pipe_path)
        loaded_pipeline = Pipeline.read().load(pipe_path)
        token_after_save = model.transform(self.data).select("token_views").take(1)[0].token_views.split("@")[2]
        lemma_after_save = model.transform(self.data).select("lemma_views").take(1)[0].lemma_views.split("@")[2]
        assert token_before_save == "sad"
        assert lemma_before_save == "unsad"
        assert token_after_save == token_before_save
        assert lemma_after_save == lemma_before_save
        pipeline_model = loaded_pipeline.fit(self.data)
        pipeline_model.transform(self.data).show()
        pipeline_model.write().overwrite().save(pipe_path)
        loaded_model = PipelineModel.read().load(pipe_path)
        loaded_model.transform(self.data).show()
        locdata = list(map(lambda d: d[0], self.data.select("text").collect()))
        spless = LightPipeline(loaded_model).annotate(locdata)
        fullSpless = LightPipeline(loaded_model).fullAnnotate(locdata)
        for row in spless[:2]:
            for _, annotations in row.items():
                for annotation in annotations[:2]:
                    print(annotation)
        for row in fullSpless[:5]:
            for _, annotations in row.items():
                for annotation in annotations[:2]:
                    print(annotation.result)
        single = LightPipeline(loaded_model).annotate("Joe was running under the rain.")
        print(single)
        assert single["lemma"][2] == "run"


class SpellCheckerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setOutputCol("token")
        spell_checker = NorvigSweetingApproach() \
            .setInputCols(["token"]) \
            .setOutputCol("spell") \
            .setDictionary("file:///" + os.getcwd() + "/../src/test/resources/spell/words.txt") \
            .setCorpus("file:///" + os.getcwd() + "/../src/test/resources/spell/sherlockholmes.txt") \
            .fit(self.data)
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.transform(assembled)
        checked = spell_checker.transform(tokenized)
        checked.show()


class SymmetricDeleteTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setOutputCol("token")
        spell_checker = SymmetricDeleteApproach() \
            .setInputCols(["token"]) \
            .setOutputCol("symmspell") \
            .setCorpus("file:///" + os.getcwd() + "/../src/test/resources/spell/sherlockholmes.txt") \
            .fit(self.data)
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.transform(assembled)
        checked = spell_checker.transform(tokenized)
        checked.show()


class ParamsGettersTestSpec(unittest.TestCase):
    @staticmethod
    def runTest():
        annotators = [DocumentAssembler, PerceptronApproach, Lemmatizer, TokenAssembler, NorvigSweetingApproach,
                      Tokenizer]
        for annotator in annotators:
            a = annotator()
            for param in a.params:
                param_name = param.name
                camelized_param = re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name)
                assert (hasattr(a, param_name))
                param_value = getattr(a, "get" + camelized_param)()
                assert (param_value is None or param_value is not None)
        # Try a getter
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence") \
            .setCustomBounds(["%%"])
        assert (sentence_detector.getOutputCol() == "sentence")
        assert (sentence_detector.getCustomBounds() == ["%%"])
        # Try a default getter
        document_assembler = DocumentAssembler()
        assert (document_assembler.getOutputCol() == "document")


class OcrTestSpec(unittest.TestCase):
    @staticmethod
    def runTest():
        data = OcrHelper.createDataset(
            spark=SparkContextForTest.spark,
            input_path="../ocr/src/test/resources/pdfs/",
            output_col="region",
            metadata_col="metadata")
        data.show()
        content = OcrHelper.createMap(input_path="../ocr/src/test/resources/pdfs/")
        print(content)
        document_assembler = DocumentAssembler() \
            .setInputCol("region") \
            .setOutputCol("document")
        document_assembler.transform(data).show()
