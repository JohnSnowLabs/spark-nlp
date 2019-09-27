from sparknlp.internal import ExtendedJavaWrapper


class NorvigSpellEvaluation(ExtendedJavaWrapper):

    def __init__(self, spark, test_file, ground_truth_file):
        ExtendedJavaWrapper.__init__(self, "com.johnsnowlabs.nlp.eval.spell.NorvigSpellEvaluation",
                                     spark._jsparkSession, test_file, ground_truth_file)

    def computeAccuracyAnnotator(self, train_file, spell):
        spell._to_java()
        return self._java_obj \
            .computeAccuracyAnnotator(train_file, spell._java_obj)

    def computeAccuracyModel(self, spell):
        return self._java_obj.computeAccuracyModel(spell._java_obj)


class SymSpellEvaluation(ExtendedJavaWrapper):

    def __init__(self, spark, test_file, ground_truth_file):
        ExtendedJavaWrapper.__init__(self, "com.johnsnowlabs.nlp.eval.spell.SymSpellEvaluation", spark._jsparkSession,
                                     test_file, ground_truth_file)

    def computeAccuracyAnnotator(self, train_file, spell):
        spell._to_java()
        return self._java_obj \
            .computeAccuracyAnnotator(train_file, spell._java_obj)

    def computeAccuracyModel(self, spell):
        return self._java_obj.computeAccuracyModel(spell._java_obj)


class NerDLEvaluation(ExtendedJavaWrapper):

    def __init__(self, spark, test_file, tag_level=""):
        ExtendedJavaWrapper.__init__(self, "com.johnsnowlabs.nlp.eval.ner.NerDLEvaluation", spark._jsparkSession,
                                     test_file, tag_level)

    def computeAccuracyModel(self, ner):
        return self._java_obj.computeAccuracyModel(ner._java_obj)

    def computeAccuracyAnnotator(self, train_file, ner, embeddings):
        ner._to_java()
        embeddings._to_java()
        return self._java_obj.computeAccuracyAnnotator(train_file, ner._java_obj, embeddings._java_obj)


class NerCrfEvaluation(ExtendedJavaWrapper):

    def __init__(self, spark, test_file, tag_level=""):
        ExtendedJavaWrapper.__init__(self, "com.johnsnowlabs.nlp.eval.ner.NerCrfEvaluation", spark._jsparkSession, test_file, tag_level)

    def computeAccuracyModel(self, ner):
        return self._java_obj.computeAccuracyModel(ner._java_obj)

    def computeAccuracyAnnotator(self, train_file, ner, embeddings):
        ner._to_java()
        embeddings._to_java()
        return self._java_obj.computeAccuracyAnnotator(train_file, ner._java_obj, embeddings._java_obj)


class POSEvaluation(ExtendedJavaWrapper):

    def __init__(self, spark, test_file):
        ExtendedJavaWrapper.__init__(self, "com.johnsnowlabs.nlp.eval.POSEvaluation", spark._jsparkSession, test_file)

    def computeAccuracyModel(self, pos):
        return self._java_obj.computeAccuracyModel(pos._java_obj)

    def computeAccuracyAnnotator(self, train_file, pos):
        pos._to_java()
        return self._java_obj.computeAccuracyAnnotator(train_file, pos._java_obj)
