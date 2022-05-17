class NormalizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark
        # self.data = SparkContextForTest.data

    def runTest(self):
        data = self.session.createDataFrame([("this is some/text I wrote",)], ["text"])
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        normalizer = Normalizer() \
            .setInputCols(["token"]) \
            .setOutputCol("normalized") \
            .setLowercase(False) \
            .setMinLength(4) \
            .setMaxLength(10)

        assembled = document_assembler.transform(data)
        tokens = tokenizer.fit(assembled).transform(assembled)
        normalizer.fit(tokens).transform(tokens).show()

