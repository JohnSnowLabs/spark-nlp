class EmbeddingsFinisherTestSpec(unittest.TestCase):

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
        glove = WordEmbeddingsModel.pretrained() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("embeddings")
        sentence_embeddings = SentenceEmbeddings() \
            .setInputCols(["sentence", "embeddings"]) \
            .setOutputCol("sentence_embeddings") \
            .setPoolingStrategy("AVERAGE")
        embeddings_finisher = EmbeddingsFinisher() \
            .setInputCols("sentence_embeddings") \
            .setOutputCols("sentence_embeddings_vectors") \
            .setOutputAsVector(True)
        explode_vectors = SQLTransformer(
            statement="SELECT EXPLODE(sentence_embeddings_vectors) AS features, * FROM __THIS__")
        kmeans = KMeans().setK(2).setSeed(1).setFeaturesCol("features")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            glove,
            sentence_embeddings,
            embeddings_finisher,
            explode_vectors,
            kmeans
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()
        # Save model
        model.write().overwrite().save("./tmp_model")
        # Load model
        PipelineModel.load("./tmp_model")

