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