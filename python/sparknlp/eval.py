from sparknlp.internal import ExtendedJavaWrapper


class NorvigSpellEvaluation(ExtendedJavaWrapper):

    def __init__(self, test_file, ground_truth_file):
        super(NorvigSpellEvaluation, self).__init__("com.johnsnowlabs.nlp.eval.NorvigSpellEvaluation")
        self._java_obj = self._new_java_obj(self._java_obj, test_file, ground_truth_file)

    def computeAccuracyAnnotator(self, train_file, spell):
        return self._java_obj.computeAccuracyAnnotator(train_file, spell)

    def testMethod(self, test_parameter):
        return self._java_obj.testMethod(test_parameter)
