class RegexTokenizerTestSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.data = SparkSessionForTest.spark.createDataFrame(
            [["AL 123456!, TX 54321-4444, AL :55555-4444, 12345-4444, 12345"]]
        ).toDF("text")

    def runTest(self) -> None:
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        pattern = "^(\\s+)|(?=[\\s+\"\'\|:;<=>!?~{}*+,$)\(&%\\[\\]])|(?<=[\\s+\"\'\|:;<=>!?~{}*+,$)\(&%\\[\\]])|(?=\.$)"

        regex_tok = RegexTokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("regex_token") \
            .setPattern(pattern) \
            .setTrimWhitespace(False) \
            .setPreservePosition(True)

        pipeline = Pipeline().setStages([document_assembler, sentence_detector, regex_tok])

        pipeline_model = pipeline.fit(self.data)

        pipe_path = "file:///" + os.getcwd() + "/tmp_regextok_pipeline"
        pipeline_model.write().overwrite().save(pipe_path)

        loaded_pipeline: PipelineModel = PipelineModel.read().load(pipe_path)
        loaded_pipeline.transform(self.data).show()

