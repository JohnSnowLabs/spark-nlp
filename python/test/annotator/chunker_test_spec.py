class ChunkerTestSpec(unittest.TestCase):

    def setUp(self):
        from sparknlp.training import POS
        self.data = SparkContextForTest.data
        self.train_pos = POS().readDataset(SparkContextForTest.spark,
                                           os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/test-training.txt",
                                           delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                           outputTextCol="text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")
        pos_tagger = PerceptronApproach() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos") \
            .setIterations(3) \
            .fit(self.train_pos)
        chunker = Chunker() \
            .setInputCols(["sentence", "pos"]) \
            .setOutputCol("chunk") \
            .setRegexParsers(["<NNP>+", "<DT|PP\\$>?<JJ>*<NN>"])
        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        pos_sentence_format = pos_tagger.transform(tokenized)
        chunk_phrases = chunker.transform(pos_sentence_format)
        chunk_phrases.show()

