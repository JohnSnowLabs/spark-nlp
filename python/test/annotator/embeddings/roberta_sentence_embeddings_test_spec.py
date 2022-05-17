class RoBertaSentenceEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_embeddings = RoBertaSentenceEmbeddings.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence_embeddings")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_embeddings
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()

