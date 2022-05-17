class EntityRulerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame([["John Snow lives in Winterfell"]]).toDF("text")
        self.path = os.getcwd() + "/../src/test/resources/entity-ruler/patterns.json"

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        entity_ruler = EntityRulerApproach() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("entity") \
            .setPatternsResource(self.path)

        pipeline = Pipeline(stages=[document_assembler, tokenizer, entity_ruler])
        model = pipeline.fit(self.data)
        model.transform(self.data).show()

