class NorvigSweetingModelTestSpec(unittest.TestCase):

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

        spell_checker = NorvigSweetingModel.pretrained() \
            .setInputCols(["token"]) \
            .setOutputCol("spell")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            spell_checker
        ])

        pipelineDF = pipeline.fit(self.data).transform(self.data)
        pipelineDF.show()

