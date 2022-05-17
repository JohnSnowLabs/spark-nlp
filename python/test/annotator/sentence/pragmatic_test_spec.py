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

