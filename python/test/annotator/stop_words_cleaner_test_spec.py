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

