import re
import unittest
import os

from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.training import *

from test.util import SparkContextForTest
from test.util import SparkSessionForTest

from pyspark.ml.feature import SQLTransformer
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import split


class BasicAnnotatorsTestSpec(unittest.TestCase):

    def setUp(self):
        # This implicitly sets up py4j for us
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token") \
            .setExceptions(["New York"]) \
            .addInfixPattern("(%\\d+)")
        stemmer = Stemmer() \
            .setInputCols(["token"]) \
            .setOutputCol("stem")
        normalizer = Normalizer() \
            .setInputCols(["stem"]) \
            .setOutputCol("normalize")
        token_assembler = TokenAssembler() \
            .setInputCols(["document", "normalize"]) \
            .setOutputCol("assembled")
        finisher = Finisher() \
            .setInputCols(["assembled"]) \
            .setOutputCols(["reassembled_view"]) \
            .setCleanAnnotations(True)
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
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
            .setInputCols(['document']) \
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
            .setInputCols(["document"]) \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma") \
            .setDictionary(path="file:///" + os.getcwd() + "/../src/test/resources/lemma-corpus-small/lemmas_small.txt",
                           key_delimiter="->", value_delimiter="\t")
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        lemmatizer.fit(tokenized).transform(tokenized).show()


class LemmatizerWithTrainingDataSetTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.conllu_file = "file:///" + os.getcwd() + "/../src/test/resources/conllu/en.test.lemma.conllu"

    def runTest(self):
        test_dataset = self.spark.createDataFrame([["So what happened?"]]).toDF("text")
        train_dataset = CoNLLU().readDataset(self.spark, self.conllu_file)
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma")

        pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer])
        pipeline.fit(train_dataset).transform(test_dataset).show()


class TokenizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark

    def runTest(self):
        data = self.session.createDataFrame([("this is some/text I wrote",)], ["text"])
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token") \
            .addInfixPattern("(\\p{L}+)(\\/)(\\p{L}+\\b)") \
            .setMinLength(3) \
            .setMaxLength(6)
        finisher = Finisher() \
            .setInputCols(["token"]) \
            .setOutputCols(["token_out"]) \
            .setOutputAsArray(True)
        assembled = document_assembler.transform(data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        finished = finisher.transform(tokenized)
        print(finished.first()['token_out'])
        self.assertEqual(len(finished.first()['token_out']), 4)


class ChunkTokenizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        entity_extractor = TextMatcher() \
            .setInputCols(['document', 'token']) \
            .setOutputCol("entity") \
            .setEntities(path="file:///" + os.getcwd() + "/../src/test/resources/entity-extractor/test-chunks.txt")
        chunk_tokenizer = ChunkTokenizer() \
            .setInputCols(['entity']) \
            .setOutputCol('chunk_token')

        pipeline = Pipeline(stages=[document_assembler, tokenizer, entity_extractor, chunk_tokenizer])

        data = self.session.createDataFrame([
            ["Hello world, my name is Michael, I am an artist and I work at Benezar"],
            ["Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week."]
        ]).toDF("text")

        pipeline.fit(data).transform(data).show()


class NormalizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        lemmatizer = Normalizer() \
            .setInputCols(["token"]) \
            .setOutputCol("normalized_token") \
            .setLowercase(False)

        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        lemmatizer.transform(tokenized).show()


class DateMatcherTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        date_matcher = DateMatcher() \
            .setInputCols(['document']) \
            .setOutputCol("date") \
            .setFormat("yyyyMM")
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
            .setInputCols(["document"]) \
            .setOutputCol("token")
        entity_extractor = TextMatcher() \
            .setInputCols(['document', 'token']) \
            .setOutputCol("entity") \
            .setEntities(path="file:///" + os.getcwd() + "/../src/test/resources/entity-extractor/test-phrases.txt")
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        entity_extractor.fit(tokenized).transform(tokenized).show()


class DocumentNormalizerSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkSessionForTest.spark.createDataFrame([
            ["""<span style="font-weight: bold; font-size: 8pt">
                     <pre style="font-family: verdana">
                      <b>The Output Y(s) of the fig. is:
                       <br /><br />
                       <img src="http://192.168.5.151/UADP4.0/ItemAuthoring/QuestionBank/Resources/94954.jpeg" />
                      </b>
                     </pre>
                    </span>"""],
            ["""<!DOCTYPE html>
                    <html>
                    <body>
                    <a class='w3schools-logo notranslate' href='//www.w3schools.com'>w3schools<span class='dotcom'>.com</span></a>
                    <h1 style="font-size:300%;">This is a heading</h1>
                    <p style="font-size:160%;">This is a paragraph containing some PII like jonhdoe@myemail.com ! John is now 42 years old.</p>
                    <p style="font-size:160%;">48% of cardiologists treated patients aged 65+.</p>
                    
                    </body>
                    </html>"""]
        ]).toDF("text")

    def runTest(self):
        df = self.data

        document_assembler = DocumentAssembler().setInputCol('text').setOutputCol('document')

        action = "clean"
        patterns = ["<[^>]*>"]
        replacement = " "
        policy = "pretty_all"

        document_normalizer = DocumentNormalizer() \
            .setInputCols("document") \
            .setOutputCol("normalizedDocument") \
            .setAction(action) \
            .setPatterns(patterns) \
            .setReplacement(replacement) \
            .setPolicy(policy) \
            .setLowercase(True)

        sentence_detector = SentenceDetector() \
            .setInputCols(["normalizedDocument"]) \
            .setOutputCol("sentence")

        regex_tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token") \
            .fit(df)

        doc_normalizer_pipeline = \
            Pipeline().setStages([document_assembler, document_normalizer, sentence_detector, regex_tokenizer])

        ds = doc_normalizer_pipeline.fit(df).transform(df)

        ds.select("normalizedDocument").show()


class PerceptronApproachTestSpec(unittest.TestCase):

    def setUp(self):
        from sparknlp.training import POS
        self.data = SparkContextForTest.data
        self.train = POS().readDataset(SparkContextForTest.spark,
                                       os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/test-training.txt",
                                       delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                       outputTextCol="text")

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
            .setIterations(1) \
            .fit(self.train)

        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        pos_tagger.transform(tokenized).show()


class ChunkerTestSpec(unittest.TestCase):

    def setUp(self):
        from sparknlp.training import POS
        self.data = SparkContextForTest.data
        self.train_pos = POS().readDataset(SparkContextForTest.spark,
                                           os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/test-training.txt",
                                           delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                           outputTextCol="text")

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
            .setIterations(3) \
            .fit(self.train_pos)
        chunker = Chunker() \
            .setInputCols(["sentence", "pos"]) \
            .setOutputCol("chunk") \
            .setRegexParsers(["<NNP>+", "<DT|PP\\$>?<JJ>*<NN>"])
        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        pos_sentence_format = pos_tagger.transform(tokenized)
        chunk_phrases = chunker.transform(pos_sentence_format)
        chunk_phrases.show()


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
            .setCustomBounds(["%%"]) \
            .setSplitLength(235) \
            .setMinLength(4) \
            .setMaxLength(50)

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
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
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
            .setInputCols(["document"]) \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma") \
            .setDictionary("file:///" + os.getcwd() + "/../src/test/resources/lemma-corpus-small/simple.txt",
                           key_delimiter="->", value_delimiter="\t")
        finisher = Finisher() \
            .setInputCols(["token", "lemma"]) \
            .setOutputCols(["token_views", "lemma_views"]) \
            .setOutputAsArray(False) \
            .setAnnotationSplitSymbol('@') \
            .setValueSplitSymbol('#')
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
        self.prediction_data = SparkContextForTest.data
        text_file = "file:///" + os.getcwd() + "/../src/test/resources/spell/sherlockholmes.txt"
        self.train_data = SparkContextForTest.spark.read.text(text_file)
        self.train_data = self.train_data.withColumnRenamed("value", "text")

    def runTest(self):

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        spell_checker = NorvigSweetingApproach() \
            .setInputCols(["token"]) \
            .setOutputCol("spell") \
            .setDictionary("file:///" + os.getcwd() + "/../src/test/resources/spell/words.txt")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            spell_checker
        ])

        model = pipeline.fit(self.train_data)
        checked = model.transform(self.prediction_data)
        checked.show()


class SymmetricDeleteTestSpec(unittest.TestCase):

    def setUp(self):
        self.prediction_data = SparkContextForTest.data
        text_file = "file:///" + os.getcwd() + "/../src/test/resources/spell/sherlockholmes.txt"
        self.train_data = SparkContextForTest.spark.read.text(text_file)
        self.train_data = self.train_data.withColumnRenamed("value", "text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        spell_checker = SymmetricDeleteApproach() \
            .setInputCols(["token"]) \
            .setOutputCol("symmspell")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            spell_checker
        ])

        model = pipeline.fit(self.train_data)
        checked = model.transform(self.prediction_data)
        checked.show()


class ContextSpellCheckerTestSpec(unittest.TestCase):

    def setUp(self):
        self.prediction_data = SparkContextForTest.data
        text_file = "file:///" + os.getcwd() + "/../src/test/resources/spell/sherlockholmes.txt"
        self.train_data = SparkContextForTest.spark.read.text(text_file)
        self.train_data = self.train_data.withColumnRenamed("value", "text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        spell_checker = ContextSpellCheckerModel \
            .pretrained('spellcheck_dl', 'en') \
            .setInputCols("token") \
            .setOutputCol("checked")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            spell_checker
        ])

        model = pipeline.fit(self.train_data)
        checked = model.transform(self.prediction_data)
        checked.show()


class ParamsGettersTestSpec(unittest.TestCase):
    @staticmethod
    def runTest():
        annotators = [DocumentAssembler, PerceptronApproach, Lemmatizer, TokenAssembler, NorvigSweetingApproach]
        for annotator in annotators:
            a = annotator()
            for param in a.params:
                param_name = param.name
                camelized_param = re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name)
                assert(hasattr(a, param_name))
                param_value = getattr(a, "get" + camelized_param)()
                assert(param_value is None or param_value is not None)
        # Try a getter
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence") \
            .setCustomBounds(["%%"])
        assert(sentence_detector.getOutputCol() == "sentence")
        assert(sentence_detector.getCustomBounds() == ["%%"])
        # Try a default getter
        document_assembler = DocumentAssembler()
        assert(document_assembler.getOutputCol() == "document")


class DependencyParserTreeBankTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark \
            .createDataFrame([["I saw a girl with a telescope"]]).toDF("text")
        self.corpus = os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/"
        self.dependency_treebank = os.getcwd() + "/../src/test/resources/parser/unlabeled/dependency_treebank"
        from sparknlp.training import POS
        self.train_pos = POS().readDataset(SparkContextForTest.spark,
                                           os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/test-training.txt",
                                           delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                           outputTextCol="text")

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
            .setIterations(1) \
            .fit(self.train_pos)

        dependency_parser = DependencyParserApproach() \
            .setInputCols(["sentence", "pos", "token"]) \
            .setOutputCol("dependency") \
            .setDependencyTreeBank(self.dependency_treebank) \
            .setNumberOfIterations(10)

        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        pos_tagged = pos_tagger.transform(tokenized)
        dependency_parsed = dependency_parser.fit(pos_tagged).transform(pos_tagged)
        dependency_parsed.show()


class DependencyParserConllUTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark \
            .createDataFrame([["I saw a girl with a telescope"]]).toDF("text")
        self.corpus = os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/"
        self.conllu = os.getcwd() + "/../src/test/resources/parser/unlabeled/conll-u/train_small.conllu.txt"
        from sparknlp.training import POS
        self.train_pos = POS().readDataset(SparkContextForTest.spark,
                                           os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/test-training.txt",
                                           delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                           outputTextCol="text")

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
            .setIterations(1) \
            .fit(self.train_pos)

        dependency_parser = DependencyParserApproach() \
            .setInputCols(["sentence", "pos", "token"]) \
            .setOutputCol("dependency") \
            .setConllU(self.conllu) \
            .setNumberOfIterations(10)

        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        pos_tagged = pos_tagger.transform(tokenized)
        dependency_parsed = dependency_parser.fit(pos_tagged).transform(pos_tagged)
        dependency_parsed.show()


class TypedDependencyParserConllUTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark \
            .createDataFrame([["I saw a girl with a telescope"]]).toDF("text")
        self.corpus = os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/"
        self.conllu = os.getcwd() + "/../src/test/resources/parser/unlabeled/conll-u/train_small.conllu.txt"
        self.conllu = os.getcwd() + "/../src/test/resources/parser/labeled/train_small.conllu.txt"
        from sparknlp.training import POS
        self.train_pos = POS().readDataset(SparkContextForTest.spark,
                                           os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/test-training.txt",
                                           delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                           outputTextCol="text")

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
            .setIterations(1) \
            .fit(self.train_pos)

        dependency_parser = DependencyParserApproach() \
            .setInputCols(["sentence", "pos", "token"]) \
            .setOutputCol("dependency") \
            .setConllU(self.conllu) \
            .setNumberOfIterations(10)

        typed_dependency_parser = TypedDependencyParserApproach() \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labdep") \
            .setConllU(self.conllu) \
            .setNumberOfIterations(10)

        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        pos_tagged = pos_tagger.transform(tokenized)
        dependency_parsed = dependency_parser.fit(pos_tagged).transform(pos_tagged)
        typed_dependency_parsed = typed_dependency_parser.fit(dependency_parsed).transform(dependency_parsed)
        typed_dependency_parsed.show()


class TypedDependencyParserConll2009TestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark \
            .createDataFrame([["I saw a girl with a telescope"]]).toDF("text")
        self.corpus = os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/"
        self.tree_bank = os.getcwd() + "/../src/test/resources/parser/unlabeled/dependency_treebank"
        self.conll2009 = os.getcwd() + "/../src/test/resources/parser/labeled/example.train.conll2009"
        from sparknlp.training import POS
        self.train_pos = POS().readDataset(SparkContextForTest.spark,
                                           os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/test-training.txt",
                                           delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                           outputTextCol="text")

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
            .setIterations(1) \
            .fit(self.train_pos)

        dependency_parser = DependencyParserApproach() \
            .setInputCols(["sentence", "pos", "token"]) \
            .setOutputCol("dependency") \
            .setDependencyTreeBank(self.tree_bank) \
            .setNumberOfIterations(10)

        typed_dependency_parser = TypedDependencyParserApproach() \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labdep") \
            .setConll2009(self.conll2009) \
            .setNumberOfIterations(10)

        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        pos_tagged = pos_tagger.transform(tokenized)
        dependency_parsed = dependency_parser.fit(pos_tagged).transform(pos_tagged)
        typed_dependency_parsed = typed_dependency_parser.fit(dependency_parsed).transform(dependency_parsed)
        typed_dependency_parsed.show()


class ChunkDocSerializingTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark \
            .createDataFrame([["I saw a girl with a telescope"]]).toDF("text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        entity_extractor = TextMatcher() \
            .setOutputCol("entity") \
            .setEntities(path="file:///" + os.getcwd() + "/../src/test/resources/entity-extractor/test-chunks.txt")
        chunk2doc = Chunk2Doc() \
            .setInputCols(['entity']) \
            .setOutputCol('entity_doc')
        doc2chunk = Doc2Chunk() \
            .setInputCols(['entity_doc']) \
            .setOutputCol('entity_rechunk')

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            entity_extractor,
            chunk2doc,
            doc2chunk
        ])

        model = pipeline.fit(self.data)
        pipe_path = "file:///" + os.getcwd() + "/tmp_chunkdoc"
        model.write().overwrite().save(pipe_path)
        PipelineModel.load(pipe_path)


class SentenceEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

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
        glove = WordEmbeddingsModel.pretrained() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("embeddings")
        sentence_embeddings = SentenceEmbeddings() \
            .setInputCols(["sentence", "embeddings"]) \
            .setOutputCol("sentence_embeddings") \
            .setPoolingStrategy("AVERAGE")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            glove,
            sentence_embeddings
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()


class StopWordsCleanerTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame([
            ["This is my first sentence. This is my second."],
            ["This is my third sentence. This is my forth."]]) \
            .toDF("text").cache()

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
        stop_words_cleaner = StopWordsCleaner() \
            .setInputCols(["token"]) \
            .setOutputCol("cleanTokens") \
            .setCaseSensitive(False) \
            .setStopWords(["this", "is"])

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            stop_words_cleaner
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).select("cleanTokens.result").show()


class StopWordsCleanerModelTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame([
            ["This is my first sentence. This is my second."],
            ["This is my third sentence. This is my forth."]]) \
            .toDF("text").cache()

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
        stop_words_cleaner = StopWordsCleaner.pretrained() \
            .setInputCols(["token"]) \
            .setOutputCol("cleanTokens") \
            .setCaseSensitive(False)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            stop_words_cleaner
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).select("cleanTokens.result").show()


class NGramGeneratorTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame([
            ["This is my first sentence. This is my second."],
            ["This is my third sentence. This is my forth."]]) \
            .toDF("text").cache()

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
        ngrams = NGramGenerator() \
            .setInputCols(["token"]) \
            .setOutputCol("ngrams") \
            .setN(2)
        ngrams_cum = NGramGenerator() \
            .setInputCols(["token"]) \
            .setOutputCol("ngrams_cum") \
            .setN(2) \
            .setEnableCumulative(True)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            ngrams,
            ngrams_cum,
        ])

        model = pipeline.fit(self.data)
        transformed_data = model.transform(self.data)
        transformed_data.select("ngrams.result", "ngrams_cum.result").show(2, False)

        assert transformed_data.select("ngrams.result").rdd.flatMap(lambda x: x).collect() == \
               [['This is', 'is my', 'my first', 'first sentence', 'sentence .', 'This is', 'is my', 'my second', 'second .'], ['This is', 'is my', 'my third', 'third sentence', 'sentence .', 'This is', 'is my', 'my forth', 'forth .']]

        assert transformed_data.select("ngrams_cum.result").rdd.flatMap(lambda x: x).collect() == \
               [['This', 'is', 'my', 'first', 'sentence', '.', 'This is', 'is my', 'my first', 'first sentence', 'sentence .', 'This', 'is', 'my', 'second', '.', 'This is', 'is my', 'my second', 'second .'], ['This', 'is', 'my', 'third', 'sentence', '.', 'This is', 'is my', 'my third', 'third sentence', 'sentence .', 'This', 'is', 'my', 'forth', '.', 'This is', 'is my', 'my forth', 'forth .']]


class ChunkEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

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
        pos_tagger = PerceptronModel.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos")
        chunker = Chunker() \
            .setInputCols(["sentence", "pos"]) \
            .setOutputCol("chunk") \
            .setRegexParsers(["<DT>?<JJ>*<NN>+"])
        glove = WordEmbeddingsModel.pretrained() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("embeddings")
        chunk_embeddings = ChunkEmbeddings() \
            .setInputCols(["chunk", "embeddings"]) \
            .setOutputCol("chunk_embeddings") \
            .setPoolingStrategy("AVERAGE")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            pos_tagger,
            chunker,
            glove,
            chunk_embeddings
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()


class EmbeddingsFinisherTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

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
        glove = WordEmbeddingsModel.pretrained() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("embeddings")
        sentence_embeddings = SentenceEmbeddings() \
            .setInputCols(["sentence", "embeddings"]) \
            .setOutputCol("sentence_embeddings") \
            .setPoolingStrategy("AVERAGE")
        embeddings_finisher = EmbeddingsFinisher() \
            .setInputCols("sentence_embeddings") \
            .setOutputCols("sentence_embeddings_vectors") \
            .setOutputAsVector(True)
        explode_vectors = SQLTransformer(statement="SELECT EXPLODE(sentence_embeddings_vectors) AS features, * FROM __THIS__")
        kmeans = KMeans().setK(2).setSeed(1).setFeaturesCol("features")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            glove,
            sentence_embeddings,
            embeddings_finisher,
            explode_vectors,
            kmeans
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()
        # Save model
        model.write().overwrite().save("./tmp_model")
        # Load model
        PipelineModel.load("./tmp_model")


class UniversalSentenceEncoderTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkSessionForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        sentence_embeddings = UniversalSentenceEncoder.pretrained() \
            .setInputCols("sentence") \
            .setOutputCol("sentence_embeddings")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            sentence_embeddings
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()


class ElmoEmbeddingsTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

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
        elmo = ElmoEmbeddings.pretrained() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("embeddings") \
            .setPoolingLayer("word_emb")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            elmo
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()


class ClassifierDLTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkSessionForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/classifier/sentiment.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_embeddings = UniversalSentenceEncoder.pretrained() \
            .setInputCols("document") \
            .setOutputCol("sentence_embeddings")

        classifier = ClassifierDLApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("label")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_embeddings,
            classifier
        ])

        model = pipeline.fit(self.data)
        model.stages[-1].write().overwrite().save('./tmp_classifierDL_model')

        classsifierdlModel = ClassifierDLModel.load("./tmp_classifierDL_model") \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCol("class")

        print(classsifierdlModel.getClasses())


class AlbertEmbeddingsTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

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
        albert = AlbertEmbeddings.pretrained() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("embeddings")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            albert
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()


class SentimentDLTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkSessionForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/classifier/sentiment.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_embeddings = UniversalSentenceEncoder.pretrained() \
            .setInputCols("document") \
            .setOutputCol("sentence_embeddings")

        classifier = SentimentDLApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("label")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_embeddings,
            classifier
        ])

        model = pipeline.fit(self.data)
        model.stages[-1].write().overwrite().save('./tmp_sentimentDL_model')

        sentimentdlModel = SentimentDLModel.load("./tmp_sentimentDL_model") \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCol("class")


class XlnetEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

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
        xlnet = XlnetEmbeddings.pretrained() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("embeddings")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            xlnet
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()


class NerDLModelTestSpec(unittest.TestCase):
    def runTest(self):
        ner_model = NerDLModel.pretrained()
        print(ner_model.getClasses())


class MultiClassifierDLTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkSessionForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/classifier/e2e.csv") \
            .withColumn("labels", split("mr", ", ")) \
            .drop("mr")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("ref") \
            .setOutputCol("document")

        sentence_embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128") \
            .setInputCols("document") \
            .setOutputCol("sentence_embeddings")

        multi_classifier = MultiClassifierDLApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("labels") \
            .setBatchSize(64) \
            .setMaxEpochs(20) \
            .setLr(0.001) \
            .setThreshold(0.5)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_embeddings,
            multi_classifier
        ])

        model = pipeline.fit(self.data)
        model.stages[-1].write().overwrite().save('./tmp_multiClassifierDL_model')

        multi_classsifierdl_model = MultiClassifierDLModel.load("./tmp_multiClassifierDL_model") \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCol("class")

        print(multi_classsifierdl_model.getClasses())


class YakeModelTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame([
            [1,"Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning "
               "competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its "
               "Cloud Next conference in San Francisco this week, the official announcement could come as early as "
               "tomorrow. Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition "
               "is happening. Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data "
               "scientists on its platform, was founded by Goldbloom  and Ben Hamner in 2010. The service got an early "
               "start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, it has managed "
               "to stay well ahead of them by focusing on its specific niche. The service is basically the de facto home "
               "for running data science and machine learning competitions. With Kaggle, Google is buying one of the "
               "largest and most active communities for data scientists - and with that, it will get increased mindshare "
               "in this community, too (though it already has plenty of that thanks to Tensorflow and other projects). "
               "Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, Google and "
               "Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. That "
               "competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that "
               "Google will keep the service running - likely under its current name. While the acquisition is probably "
               "more about Kaggle's community than technology, Kaggle did build some interesting tools for hosting its "
               "competition and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets "
               "and developers can share this code on the platform (the company previously called them 'scripts'). Like "
               "similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do "
               "with that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook "
               "says it's $12.75) since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max "
               "Levchin, Naval Ravikant, Google chief economist Hal Varian, Khosla Ventures and Yuri Milner"]
        ]).toDF("id", "text").cache()

    def runTest(self):
        document = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence = SentenceDetector() \
            .setInputCols("document") \
            .setOutputCol("sentence")

        token = Tokenizer() \
            .setInputCols("sentence") \
            .setOutputCol("token") \
            .setContextChars(["(", ")", "?", "!", ".", ","])

        keywords = YakeModel() \
            .setInputCols("token") \
            .setOutputCol("keywords") \
            .setMinNGrams(2) \
            .setMaxNGrams(3)
        pipeline = Pipeline(stages=[document, sentence, token, keywords])

        result = pipeline.fit(self.data).transform(self.data)
        result.select("keywords").show(truncate=False)


class SentenceDetectorDLTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetectorDLModel.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()


class WordSegmenterTestSpec(unittest.TestCase):

    def setUp(self):
        from sparknlp.training import POS
        self.data = SparkContextForTest.spark.createDataFrame([["十四不是四十"]]) \
            .toDF("text").cache()
        self.train = POS().readDataset(SparkContextForTest.spark,
                                       os.getcwd() + "/../src/test/resources/word-segmenter/chinese_train.utf8",
                                       delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                       outputTextCol="text")
    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        word_segmenter = WordSegmenterApproach() \
            .setInputCols("document") \
            .setOutputCol("token") \
            .setPosCol("tags") \
            .setIterations(1) \
            .fit(self.train)
        pipeline = Pipeline(stages=[
            document_assembler,
            word_segmenter
        ])

        model = pipeline.fit(self.train)
        model.transform(self.data).show(truncate=False)                                   

class LanguageDetectorDLTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.read \
            .option("delimiter", "|") \
            .option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/language-detector/multilingual_sample.txt")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        
        sentence_detector = SentenceDetectorDLModel.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        ld = LanguageDetectorDL.pretrained()

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            ld
        ])

        # list all the languages
        print(ld.getLanguages())

        model = pipeline.fit(self.data)
        model.transform(self.data).show()


class T5TransformerQATestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, "Which is the capital of France? Who was the first president of USA?"],
            [1, "Which is the capital of Bulgaria ?"],
            [2, "Who is Donald Trump?"]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        sentence_detector = SentenceDetectorDLModel \
            .pretrained() \
            .setInputCols(["documents"]) \
            .setOutputCol("questions")

        t5 = T5Transformer.pretrained() \
            .setInputCols(["questions"]) \
            .setOutputCol("answers")

        pipeline = Pipeline().setStages([document_assembler, sentence_detector, t5])
        results = pipeline.fit(data).transform(data)

        results.select("questions.result", "answers.result").show(truncate=False)


class T5TransformerSummaryTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, """
            Heat oven to 200C/180C fan/gas 6. Line each hole of a 12-hole muffin tin with a thin strip of baking 
            parchment across the middle that’s long enough so the ends stick out a centimetre or two – use a dab of
             butter to stick in place. Roll out two thirds of the pastry on a lightly floured surface and stamp out 
             12 x 10cm circles (you may need to re-roll trimmings). Press a circle into each hole to line.
             
            Sprinkle 1 tsp of breadcrumbs into the base of each pie. Tip the rest of the crumbs into a mixing bowl. 
            Squeeze in the sausage meat, discarding the skins, along with the bacon, mace, pepper, sage and just a 
            little salt. Get your hands in and mash and squish everything together until the breadcrumbs have just 
            about disappeared. Divide mixture between the holes, packing in firmly and shaping to a dome in the middle.
             
            Roll out the remaining pastry and stamp out 12 x 7cm circles. Brush with a little egg and add a top to 
            each pie, egg-side down to stick, carefully pressing pastry edges together to seal. Brush with more egg 
            (don’t throw away leftovers) and sprinkle with sesame seeds. Bake for 30 mins until golden then carefully 
            remove the pies from the tin, using the parchment ends to help you lift them out. Sit on a parchment lined 
            baking tray, brush all round the sides with more egg and put back in the oven for 8 mins. Cool completely 
            then eat with piccalilli, or your favourite pickle.             
            """.strip().replace("\n", " ")]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        t5 = T5Transformer.pretrained()\
            .setTask("summarize:") \
            .setMaxOutputLength(200) \
            .setInputCols(["documents"]) \
            .setOutputCol("summaries")

        pipeline = Pipeline().setStages([document_assembler, t5])
        results = pipeline.fit(data).transform(data)

        results.select("summaries.result").show(truncate=False)

