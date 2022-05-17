class ChunkTokenizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark

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
            .setEntities(path="file:///" + os.getcwd() + "/../src/test/resources/entity-extractor/test-chunks.txt")
        chunk_tokenizer = ChunkTokenizer() \
            .setInputCols(['entity']) \
            .setOutputCol('chunk_token')

        pipeline = Pipeline(stages=[document_assembler, tokenizer, entity_extractor, chunk_tokenizer])

        data = self.session.createDataFrame([
            ["Hello world, my name is Michael, I am an artist and I work at Benezar"],
            ["Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week."]
        ]).toDF("text")

        pipeline.fit(data).transform(data).show()

