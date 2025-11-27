#  Copyright 2017-2022 John Snow Labs
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
from sparknlp.annotator.classifier_dl import BertForQuestionAnswering


class TapasForQuestionAnswering(BertForQuestionAnswering):
    """TapasForQuestionAnswering is an implementation of TaPas - a BERT-based model specifically designed for
    answering questions about tabular data. It takes TABLE and DOCUMENT annotations as input and tries to answer
    the questions in the document by using the data from the table. The model is based in BertForQuestionAnswering
    and shares all its parameters with it.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> tapas = TapasForQuestionAnswering.pretrained() \\
    ...     .setInputCols(["table", "document"]) \\
    ...     .setOutputCol("answer")

    The default model is ``"table_qa_tapas_base_finetuned_wtq"``, if no name
    is provided.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Question+Answering+Tapas>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TABLE``    ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 2
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        False
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    maxSentenceLength
        Max sentence length to process, by default 512

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>>
    >>> document_assembler = MultiDocumentAssembler()\\
    ...     .setInputCols("table_json", "questions")\\
    ...     .setOutputCols("document_table", "document_questions")
    >>>
    >>> sentence_detector = SentenceDetector()\\
    ...     .setInputCols(["document_questions"])\\
    ...     .setOutputCol("questions")
    >>>
    >>> table_assembler = TableAssembler()\\
    ...     .setInputCols(["document_table"])\\
    ...     .setOutputCol("table")
    >>>
    >>> tapas = TapasForQuestionAnswering\\
    ...     .pretrained()\\
    ...     .setInputCols(["questions", "table"])\\
    ...     .setOutputCol("answers")
    >>>
    >>> pipeline = Pipeline(stages=[
    ...     document_assembler,
    ...     sentence_detector,
    ...     table_assembler,
    ...     tapas])
    >>>
    >>> json_data = \"\"\"
    ... {
    ...     "header": ["name", "money", "age"],
    ...     "rows": [
    ...     ["Donald Trump", "$100,000,000", "75"],
    ...     ["Elon Musk", "$20,000,000,000,000", "55"]
    ...     ]
    ...  }
    ...  \"\"\"
    >>> model = pipeline.fit(data)
    >>> model\\
    ...     .transform(data)\\
    ...     .selectExpr("explode(answers) AS answer")\\
    ...     .select("answer.metadata.question", "answer.result")\\
    ...     .show(truncate=False)
    +-----------------------+----------------------------------------+
    |question               |result                                  |
    +-----------------------+----------------------------------------+
    |Who earns 100,000,000? |Donald Trump                            |
    |Who has more money?    |Elon Musk                               |
    |How much they all earn?|COUNT($100,000,000, $20,000,000,000,000)|
    |How old are they?      |AVERAGE(75, 55)                         |
    +-----------------------+----------------------------------------+
    """

    name = "TapasForQuestionAnswering"

    inputAnnotatorTypes = [AnnotatorType.TABLE, AnnotatorType.DOCUMENT]

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.TapasForQuestionAnswering",
                 java_model=None):
        super(TapasForQuestionAnswering, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=2,
            maxSentenceLength=512,
            caseSensitive=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        TapasForQuestionAnswering
            The restored model
        """
        from sparknlp.internal import _TapasForQuestionAnsweringLoader
        jModel = _TapasForQuestionAnsweringLoader(folder, spark_session._jsparkSession)._java_obj
        return TapasForQuestionAnswering(java_model=jModel)

    @staticmethod
    def pretrained(name="table_qa_tapas_base_finetuned_wtq", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "table_qa_tapas_base_finetuned_wtq"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        TapasForQuestionAnswering
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TapasForQuestionAnswering, name, lang, remote_loc)
