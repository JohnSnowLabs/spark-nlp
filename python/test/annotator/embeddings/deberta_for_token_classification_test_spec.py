class DeBertaForTokenClassificationTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        doc_classifier = DeBertaForTokenClassification \
            .pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("class")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            doc_classifier
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()
