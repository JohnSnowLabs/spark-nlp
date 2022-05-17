class ChunkEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")
        pos_tagger = PerceptronModel.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos")
        chunker = Chunker() \
            .setInputCols(["sentence", "pos"]) \
            .setOutputCol("chunk") \
            .setRegexParsers(["<DT>?<JJ>*<NN>+"])
        glove = WordEmbeddingsModel.pretrained() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("embeddings")
        chunk_embeddings = ChunkEmbeddings() \
            .setInputCols(["chunk", "embeddings"]) \
            .setOutputCol("chunk_embeddings") \
            .setPoolingStrategy("AVERAGE")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            pos_tagger,
            chunker,
            glove,
            chunk_embeddings
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()

