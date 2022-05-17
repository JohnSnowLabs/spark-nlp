class NerDLModelTestSpec(unittest.TestCase):
    def runTest(self):
        ner_model = NerDLModel.pretrained()
        print(ner_model.getClasses())

