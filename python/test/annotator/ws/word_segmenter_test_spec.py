class WordSegmenterTestSpec(unittest.TestCase):

    def setUp(self):
        from sparknlp.training import POS
        self.data = SparkContextForTest.spark.createDataFrame([["十四不是四十"]]) \
            .toDF("text").cache()
        self.train = POS().readDataset(SparkContextForTest.spark,
                                       os.getcwd() + "/../src/test/resources/word-segmenter/chinese_train.utf8",
                                       delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                       outputTextCol="text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        word_segmenter = WordSegmenterApproach() \
            .setInputCols("document") \
            .setOutputCol("token") \
            .setPosColumn("tags") \
            .setNIterations(1) \
            .fit(self.train)
        pipeline = Pipeline(stages=[
            document_assembler,
            word_segmenter
        ])

        model = pipeline.fit(self.train)
        model.transform(self.data).show(truncate=False)

