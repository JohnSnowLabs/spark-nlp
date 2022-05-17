class SentimentDLTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkSessionForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/classifier/sentiment.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_embeddings = UniversalSentenceEncoder.pretrained() \
            .setInputCols("document") \
            .setOutputCol("sentence_embeddings")

        classifier = SentimentDLApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("label") \
            .setRandomSeed(44)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_embeddings,
            classifier
        ])

        model = pipeline.fit(self.data)
        model.stages[-1].write().overwrite().save('./tmp_sentimentDL_model')

        sentimentdlModel = SentimentDLModel.load("./tmp_sentimentDL_model") \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCol("class")

