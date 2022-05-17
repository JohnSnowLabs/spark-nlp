class GetClassesTestSpec(unittest.TestCase):

    def runTest(self):
        print(AlbertForTokenClassification.pretrained().getClasses())
        print(XlnetForTokenClassification.pretrained().getClasses())
        print(BertForTokenClassification.pretrained().getClasses())
        print(DistilBertForTokenClassification.pretrained().getClasses())
        print(RoBertaForTokenClassification.pretrained().getClasses())
        print(XlmRoBertaForTokenClassification.pretrained().getClasses())
        print(LongformerForTokenClassification.pretrained().getClasses())

        print(AlbertForSequenceClassification.pretrained().getClasses())
        print(XlnetForSequenceClassification.pretrained().getClasses())
        print(BertForSequenceClassification.pretrained().getClasses())
        print(DistilBertForSequenceClassification.pretrained().getClasses())
        print(RoBertaForSequenceClassification.pretrained().getClasses())
        print(XlmRoBertaForSequenceClassification.pretrained().getClasses())
        print(LongformerForSequenceClassification.pretrained().getClasses())

