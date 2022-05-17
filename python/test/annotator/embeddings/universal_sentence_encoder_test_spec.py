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

