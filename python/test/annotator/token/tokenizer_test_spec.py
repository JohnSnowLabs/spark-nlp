class TokenizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark

    def runTest(self):
        data = self.session.createDataFrame([("this is some/text I wrote",)], ["text"])
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token") \
            .addInfixPattern("(\\p{L}+)(\\/)(\\p{L}+\\b)") \
            .setMinLength(3) \
            .setMaxLength(6)
        finisher = Finisher() \
            .setInputCols(["token"]) \
            .setOutputCols(["token_out"]) \
            .setOutputAsArray(True)
        assembled = document_assembler.transform(data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        finished = finisher.transform(tokenized)
        print(finished.first()['token_out'])
        self.assertEqual(len(finished.first()['token_out']), 4)

class TokenizerWithExceptionsTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark

    def runTest(self):
        data = self.session.createDataFrame(
            [("My friend moved to New York. She likes it. Frank visited New York, and didn't like it.",)], ["text"])
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token") \
            .setExceptionsPath(path="file:///" + os.getcwd() + "/../src/test/resources/token_exception_list.txt")
        finisher = Finisher() \
            .setInputCols(["token"]) \
            .setOutputCols(["token_out"]) \
            .setOutputAsArray(True)
        assembled = document_assembler.transform(data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        finished = finisher.transform(tokenized)
        # print(finished.first()['token_out'])
        self.assertEqual((finished.first()['token_out']).index("New York."), 4)
        self.assertEqual((finished.first()['token_out']).index("New York,"), 11)

