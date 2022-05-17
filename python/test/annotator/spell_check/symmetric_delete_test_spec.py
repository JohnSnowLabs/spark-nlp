class SymmetricDeleteTestSpec(unittest.TestCase):

    def setUp(self):
        self.prediction_data = SparkContextForTest.data
        text_file = "file:///" + os.getcwd() + "/../src/test/resources/spell/sherlockholmes.txt"
        self.train_data = SparkContextForTest.spark.read.text(text_file)
        self.train_data = self.train_data.withColumnRenamed("value", "text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        spell_checker = SymmetricDeleteApproach() \
            .setInputCols(["token"]) \
            .setOutputCol("symmspell")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            spell_checker
        ])

        model = pipeline.fit(self.train_data)
        checked = model.transform(self.prediction_data)
        checked.show()

