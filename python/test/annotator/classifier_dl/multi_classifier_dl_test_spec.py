class MultiClassifierDLTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkSessionForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/classifier/e2e.csv") \
            .withColumn("labels", split("mr", ", ")) \
            .drop("mr")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("ref") \
            .setOutputCol("document")

        sentence_embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128") \
            .setInputCols("document") \
            .setOutputCol("sentence_embeddings")

        multi_classifier = MultiClassifierDLApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("labels") \
            .setBatchSize(64) \
            .setMaxEpochs(20) \
            .setLr(0.001) \
            .setThreshold(0.5) \
            .setRandomSeed(44)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_embeddings,
            multi_classifier
        ])

        model = pipeline.fit(self.data)
        model.stages[-1].write().overwrite().save('./tmp_multiClassifierDL_model')

        multi_classsifierdl_model = MultiClassifierDLModel.load("./tmp_multiClassifierDL_model") \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCol("class")

        print(multi_classsifierdl_model.getClasses())

