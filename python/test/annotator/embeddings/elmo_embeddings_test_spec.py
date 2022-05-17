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

