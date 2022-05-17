class GPT2TransformerTextGenerationTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, """Leonardo Da Vinci invented the microscope?""".strip().replace("\n", " ")]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        gpt2 = GPT2Transformer \
            .pretrained() \
            .setTask("is it true that") \
            .setMaxOutputLength(50) \
            .setDoSample(True) \
            .setInputCols(["documents"]) \
            .setOutputCol("generation")

        pipeline = Pipeline().setStages([document_assembler, gpt2])
        results = pipeline.fit(data).transform(data)

        results.select("generation.result").show(truncate=False)

