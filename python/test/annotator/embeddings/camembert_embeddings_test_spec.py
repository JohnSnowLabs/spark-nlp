class CamemBertEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        embeddings = CamemBertEmbeddings.pretrained() \
            .setInputCols(["token", "document"]) \
            .setOutputCol("camembert_embeddings")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            embeddings
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()
