from sparknlp.internal import ExtendedJavaWrapper


class NorvigSpellEvaluation(ExtendedJavaWrapper):

    def __init__(self, test_file, ground_truth_file):
        ExtendedJavaWrapper.__init__(self, "com.johnsnowlabs.nlp.eval.spell.NorvigSpellEvaluation")
        self._java_obj = self._new_java_obj(self._java_obj, test_file, ground_truth_file)

    def computeAccuracyAnnotator(self, train_file, spell):
        input_cols = spell.getInputCols()
        output_col = spell.getOutputCol()
        java_input_cols = self.new_java_array_string(input_cols)
        return self._java_obj.computeAccuracyAnnotator(train_file, java_input_cols, output_col, spell.dictionary_path)

    def computeAccuracyModel(self, spell):
        return self._java_obj.computeAccuracyModel(spell._java_obj)


class SymSpellEvaluation(ExtendedJavaWrapper):

    def __init__(self, test_file, ground_truth_file):
        ExtendedJavaWrapper.__init__(self, "com.johnsnowlabs.nlp.eval.spell.SymSpellEvaluation")
        self._java_obj = self._new_java_obj(self._java_obj, test_file, ground_truth_file)

    def computeAccuracyAnnotator(self, train_file, spell):
        input_cols = spell.getInputCols()
        output_col = spell.getOutputCol()
        java_input_cols = self.new_java_array_string(input_cols)
        return self._java_obj.computeAccuracyAnnotator(train_file, java_input_cols, output_col, spell.dictionary_path)

    def computeAccuracyModel(self, spell):
        return self._java_obj.computeAccuracyModel(spell._java_obj)


class NerDLEvaluation(ExtendedJavaWrapper):

    def __init__(self, spark, test_file, tag_level):
        ExtendedJavaWrapper.__init__(self, "com.johnsnowlabs.nlp.eval.ner.NerDLEvaluation")
        self._java_obj = self._new_java_obj(self._java_obj, spark._jsparkSession, test_file, tag_level)

    def computeAccuracyModel(self, ner):
        return self._java_obj.computeAccuracyModel(ner._java_obj)

    def computeAccuracyAnnotator(self, train_file, ner, embeddings):
        ner_input_cols = ner.getInputCols()
        ner_output_col = ner.getOutputCol()
        label_column = ner.getLabelColumn()
        ner_java_input_cols = self.new_java_array_string(ner_input_cols)
        embeddings_input_cols = embeddings.getInputCols()
        embeddings_output_col = embeddings.getOutputCol()
        embeddings_java_input_cols = self.new_java_array_string(embeddings_input_cols)
        random_seed = ner.getRandomSeed()
        embeddings_path = embeddings.getEmbeddingsPath()
        embeddings_dimension = embeddings.getDimension()
        embeddings_format = embeddings.getFormat()
        return self._java_obj.computeAccuracyAnnotator(train_file, ner_java_input_cols, ner_output_col, random_seed,
                                                       label_column,
                                                       embeddings_java_input_cols, embeddings_output_col,
                                                       embeddings_path, embeddings_dimension, embeddings_format)


class NerCrfEvaluation(ExtendedJavaWrapper):

    def __init__(self, spark, test_file, tag_level):
        ExtendedJavaWrapper.__init__(self, "com.johnsnowlabs.nlp.eval.ner.NerCrfEvaluation")
        self._java_obj = self._new_java_obj(self._java_obj, spark._jsparkSession, test_file, tag_level)

    def computeAccuracyModel(self, ner):
        return self._java_obj.computeAccuracyModel(ner._java_obj)

    def computeAccuracyAnnotator(self, train_file, ner, embeddings):
        ner_input_cols = ner.getInputCols()
        ner_output_col = ner.getOutputCol()
        label_column = ner.getLabelColumn()
        ner_java_input_cols = self.new_java_array_string(ner_input_cols)
        embeddings_input_cols = embeddings.getInputCols()
        embeddings_output_col = embeddings.getOutputCol()
        embeddings_java_input_cols = self.new_java_array_string(embeddings_input_cols)
        random_seed = ner.getRandomSeed()
        embeddings_path = embeddings.getEmbeddingsPath()
        embeddings_dimension = embeddings.getDimension()
        embeddings_format = embeddings.getFormat()
        return self._java_obj.computeAccuracyAnnotator(train_file, ner_java_input_cols, ner_output_col, random_seed,
                                                       label_column,
                                                       embeddings_java_input_cols, embeddings_output_col,
                                                       embeddings_path, embeddings_dimension, embeddings_format)
