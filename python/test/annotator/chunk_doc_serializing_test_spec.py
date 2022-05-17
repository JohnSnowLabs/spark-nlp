class ChunkDocSerializingTestSpec(unittest.TestCase):
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
        entity_extractor = TextMatcher() \
            .setOutputCol("entity") \
            .setEntities(path="file:///" + os.getcwd() + "/../src/test/resources/entity-extractor/test-chunks.txt")
        chunk2doc = Chunk2Doc() \
            .setInputCols(['entity']) \
            .setOutputCol('entity_doc')
        doc2chunk = Doc2Chunk() \
            .setInputCols(['entity_doc']) \
            .setOutputCol('entity_rechunk')

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            entity_extractor,
            chunk2doc,
            doc2chunk
        ])

        model = pipeline.fit(self.data)
        pipe_path = "file:///" + os.getcwd() + "/tmp_chunkdoc"
        model.write().overwrite().save(pipe_path)
        PipelineModel.load(pipe_path)

