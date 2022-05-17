class DateMatcherTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        date_matcher = DateMatcher() \
            .setInputCols(['document']) \
            .setOutputCol("date") \
            .setOutputFormat("yyyyMM") \
            .setSourceLanguage("en")

        assembled = document_assembler.transform(self.data)
        date_matcher.transform(assembled).show()

