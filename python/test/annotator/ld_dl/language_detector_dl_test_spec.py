class LanguageDetectorDLTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.read \
            .option("delimiter", "|") \
            .option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/language-detector/multilingual_sample.txt")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetectorDLModel.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        ld = LanguageDetectorDL.pretrained()

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            ld
        ])

        # list all the languages
        print(ld.getLanguages())

        model = pipeline.fit(self.data)
        model.transform(self.data).show()

