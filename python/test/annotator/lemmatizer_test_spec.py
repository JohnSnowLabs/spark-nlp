class LemmatizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma") \
            .setDictionary(path="file:///" + os.getcwd() + "/../src/test/resources/lemma-corpus-small/lemmas_small.txt",
                           key_delimiter="->", value_delimiter="\t")
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        lemmatizer.fit(tokenized).transform(tokenized).show()

class LemmatizerWithTrainingDataSetTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.conllu_file = "file:///" + os.getcwd() + "/../src/test/resources/conllu/en.test.lemma.conllu"

    def runTest(self):
        test_dataset = self.spark.createDataFrame([["So what happened?"]]).toDF("text")
        train_dataset = CoNLLU(lemmaCol="lemma_train").readDataset(self.spark, self.conllu_file)
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setFormCol("form") \
            .setLemmaCol("lemma_train") \
            .setOutputCol("lemma")

        train_dataset.show()
        pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer])
        pipeline.fit(train_dataset).transform(test_dataset).show()

