class TextMatcherTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        entity_extractor = TextMatcher() \
            .setInputCols(['document', 'token']) \
            .setOutputCol("entity") \
            .setEntities(path="file:///" + os.getcwd() + "/../src/test/resources/entity-extractor/test-phrases.txt")
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        entity_extractor.fit(tokenized).transform(tokenized).show()

