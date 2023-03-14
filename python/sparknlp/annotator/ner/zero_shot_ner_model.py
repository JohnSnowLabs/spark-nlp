#  Copyright 2017-2023 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from sparknlp.common import *
from sparknlp.annotator.classifier_dl import RoBertaForQuestionAnswering


class ZeroShotNerModel(RoBertaForQuestionAnswering, HasEngine):
    """ZeroShotNerModel implements zero shot named entity recognition by utilizing RoBERTa
    transformer models fine tuned on a question answering task.

    Its input is a list of document annotations and it automatically generates questions which are
    used to recognize entities. The definitions of entities is given by a dictionary structures,
    specifying a set of questions for each entity. The model is based on
    RoBertaForQuestionAnswering.

    For more extended examples see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/named-entity-recognition/ZeroShot_NER.ipynb>`__.

    Pretrained models can be loaded with ``pretrained`` of the companion object:

    .. code-block:: python

       zeroShotNer = ZeroShotNerModel.pretrained() \\
           .setInputCols(["document"]) \\
           .setOutputCol("zer_shot_ner")

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``NAMED_ENTITY``
    ====================== ======================

    Parameters
    ----------
    entityDefinitions
        A dictionary with definitions of named entities. The keys of dictionary are the entity labels and the
        values are lists of questions. For example:
              {
                "CITY": ["Which city?", "Which town?"],
                "NAME": ["What is her name?", "What is his name?"]}

    predictionThreshold
        Minimal confidence score to encode an entity (Default: 0.01f)
    ignoreEntities
        A list of entity labels which are discarded from the output.

    References
    ----------
    `RoBERTa: A Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`__ : for details about the RoBERTa transformer
    :class:`.RoBertaForQuestionAnswering` : for the SparkNLP implementation of RoBERTa question  answering

    Examples
    --------
    >>> document_assembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence_detector = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> zero_shot_ner = ZeroShotNerModel() \\
    ...     .pretrained() \\
    ...     .setEntityDefinitions(
    ...         {
    ...             "NAME": ["What is his name?", "What is my name?", "What is her name?"],
    ...             "CITY": ["Which city?", "Which is the city?"]
    ...         }) \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("zero_shot_ner") \\
    >>> data = spark.createDataFrame(
    ...         [["My name is Clara, I live in New York and Hellen lives in Paris."]]
    ...     ).toDF("text")
    >>> Pipeline() \\
    ...     .setStages([document_assembler, sentence_detector, tokenizer, zero_shot_ner]) \\
    ...     .fit(data) \\
    ...     .transform(data) \\
    ...     .selectExpr("document", "explode(zero_shot_ner) AS entity") \\
    ...     .select(
    ...         "document.result",
    ...         "entity.result",
    ...         "entity.metadata.word",
    ...         "entity.metadata.confidence",
    ...         "entity.metadata.question") \\
    ...     .show(truncate=False)
    +-----------------------------------------------------------------+------+------+----------+------------------+
    |result                                                           |result|word  |confidence|question          |
    +-----------------------------------------------------------------+------+------+----------+------------------+
    |[My name is Clara, I live in New York and Hellen lives in Paris.]|B-CITY|Paris |0.5328949 |Which is the city?|
    |[My name is Clara, I live in New York and Hellen lives in Paris.]|B-NAME|Clara |0.9360068 |What is my name?  |
    |[My name is Clara, I live in New York and Hellen lives in Paris.]|B-CITY|New   |0.83294415|Which city?       |
    |[My name is Clara, I live in New York and Hellen lives in Paris.]|I-CITY|York  |0.83294415|Which city?       |
    |[My name is Clara, I live in New York and Hellen lives in Paris.]|B-NAME|Hellen|0.45366877|What is her name? |
    +-----------------------------------------------------------------+------+------+----------+------------------+
    """
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]
    outputAnnotatorType = AnnotatorType.NAMED_ENTITY

    name = "ZeroShotNerModel"

    predictionThreshold = Param(Params._dummy(),
                                "predictionThreshold",
                                "Minimal confidence score to encode an entity (default is 0.1)",
                                TypeConverters.toFloat)

    ignoreEntities = Param(Params._dummy(),
                           "ignoreEntities",
                           "List of entities to ignore",
                           TypeConverters.toListString)

    def setPredictionThreshold(self, threshold):
        """Sets the minimal confidence score to encode an entity

        Parameters
        ----------
        threshold : float
           minimal confidence score to encode an entity (default is 0.1)
        """
        return self._set(predictionThreshold=threshold)

    def setEntityDefinitions(self, definitions):
        """Set entity definitions

        Parameters
        ----------
        definitions : dict[str, list[str]]

        """
        self._call_java("setEntityDefinitions", definitions)

        return self

    def getClasses(self):
        """
        Returns the list of entities which are recognized
        """
        return self._call_java("getEntities")

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.dl.ZeroShotNerModel", java_model=None):
        super(ZeroShotNerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            predictionThreshold=0.01,
            ignoreEntities=[]
        )

    @staticmethod
    def pretrained(name="zero_shot_ner_roberta", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ZeroShotNerModel, name, lang, remote_loc,
                                                j_dwn='PythonResourceDownloader')

    @staticmethod
    def load(path):
        from sparknlp.internal import _RobertaQAToZeroShotNerLoader
        jModel = _RobertaQAToZeroShotNerLoader(path)._java_obj
        return ZeroShotNerModel(java_model=jModel)
