class GraphExtractionTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.data_set = self.spark.createDataFrame([["Peter Parker is a nice person and lives in New York"]]).toDF(
            "text")

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        word_embeddings = WordEmbeddingsModel.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("embeddings")

        ner_model = NerDLModel.pretrained() \
            .setInputCols(["document", "token", "embeddings"]) \
            .setOutputCol("ner")

        pos_tagger = PerceptronModel.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("pos")

        dependency_parser = DependencyParserModel.pretrained() \
            .setInputCols(["document", "pos", "token"]) \
            .setOutputCol("dependency")

        typed_dependency_parser = TypedDependencyParserModel.pretrained() \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labdep")

        graph_extraction = GraphExtraction() \
            .setInputCols(["document", "token", "ner"]) \
            .setOutputCol("graph") \
            .setRelationshipTypes(["person-PER", "person-LOC"])

        graph_finisher = GraphFinisher() \
            .setInputCol("graph") \
            .setOutputCol("finisher")

        pipeline = Pipeline().setStages([document_assembler, tokenizer,
                                         word_embeddings, ner_model, pos_tagger,
                                         dependency_parser, typed_dependency_parser])

        test_data_set = pipeline.fit(self.data_set).transform(self.data_set)
        pipeline_finisher = Pipeline().setStages([graph_extraction, graph_finisher])

        graph_data_set = pipeline_finisher.fit(test_data_set).transform(test_data_set)
        graph_data_set.show(truncate=False)

