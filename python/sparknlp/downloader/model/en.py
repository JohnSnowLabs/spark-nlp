from sparknlp.downloader import ResourceDownloader
from sparknlp.annotator import *


class CloudPerceptronModel:
    @staticmethod
    def retrieve():
        return ResourceDownloader.downloadModel(PerceptronModel, "pos_fast", "en")


class CloudNerCrfModel:
    @staticmethod
    def retrieve():
        return ResourceDownloader.downloadModel(NerCrfModel, "ner_fast", "en")


class CloudLemmatizer:
    @staticmethod
    def retrieve():
        return ResourceDownloader.downloadModel(LemmatizerModel, "lemma_fast", "en")


class CloudSpellChecker:
    @staticmethod
    def retrieve():
        return ResourceDownloader.downloadModel(NorvigSweetingModel, "spell_fast", "en")


class CloudViveknSentiment:
    @staticmethod
    def retrieve():
        return ResourceDownloader.downloadModel(ViveknSentimentModel, "vivekn_fast", "en")
