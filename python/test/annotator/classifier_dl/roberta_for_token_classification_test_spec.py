class RoBertaForTokenClassificationTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        token_classifier = RoBertaForTokenClassification.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("ner")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            token_classifier
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()

