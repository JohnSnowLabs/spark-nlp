class ParamsGettersTestSpec(unittest.TestCase):
    @staticmethod
    def runTest():
        annotators = [DocumentAssembler, PerceptronApproach, Lemmatizer, TokenAssembler, NorvigSweetingApproach]
        for annotator in annotators:
            a = annotator()
            for param in a.params:
                param_name = param.name
                camelized_param = re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name)
                assert (hasattr(a, param_name))
                param_value = getattr(a, "get" + camelized_param)()
                assert (param_value is None or param_value is not None)
        # Try a getter
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence") \
            .setCustomBounds(["%%"])
        assert (sentence_detector.getOutputCol() == "sentence")
        assert (sentence_detector.getCustomBounds() == ["%%"])
        # Try a default getter
        document_assembler = DocumentAssembler()
        assert (document_assembler.getOutputCol() == "document")

