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
"""Contains classes for CoNLL."""

import pyspark

from sparknlp.common import ReadAs
from sparknlp.internal import ExtendedJavaWrapper


class CoNLL(ExtendedJavaWrapper):
    """Instantiates the class to read a CoNLL dataset.

    The dataset should be in the format of `CoNLL 2003
    <https://www.clips.uantwerpen.be/conll2003/ner/>`_ and needs to be specified
    with :meth:`.readDataset`, which will create a dataframe with the data.

    Can be used to train a :class:`NerDLApproach
    <sparknlp.annotator.NerDLApproach>`.

    **Input File Format**::

        -DOCSTART- -X- -X- O

        EU NNP B-NP B-ORG
        rejects VBZ B-VP O
        German JJ B-NP B-MISC
        call NN I-NP O
        to TO B-VP O
        boycott VB I-VP O
        British JJ B-NP B-MISC
        lamb NN I-NP O
        . . O O

    Parameters
    ----------
    documentCol : str, optional
        Name of the :class:`.DocumentAssembler` column, by default 'document'
    sentenceCol : str, optional
        Name of the :class:`.SentenceDetector` column, by default 'sentence'
    tokenCol : str, optional
        Name of the :class:`.Tokenizer` column, by default 'token'
    posCol : str, optional
        Name of the :class:`.PerceptronModel` column, by default 'pos'
    conllLabelIndex : int, optional
        Index of the label column in the dataset, by default 3
    conllPosIndex : int, optional
        Index of the POS tags in the dataset, by default 1
    textCol : str, optional
        Index of the text column in the dataset, by default 'text'
    labelCol : str, optional
        Name of the label column, by default 'label'
    explodeSentences : bool, optional
        Whether to explode sentences to separate rows, by default True
    delimiter: str, optional
        Delimiter used to separate columns inside CoNLL file
    includeDocId: bool, optional
        Whether to try and parse the document id from the third item in the -DOCSTART- line (X if not found)

    Examples
    --------
    >>> from sparknlp.training import CoNLL
    >>> trainingData = CoNLL().readDataset(spark, "src/test/resources/conll2003/eng.train")
    >>> trainingData.selectExpr(
    ...     "text",
    ...     "token.result as tokens",
    ...     "pos.result as pos",
    ...     "label.result as label"
    ... ).show(3, False)
    +------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
    |text                                            |tokens                                                    |pos                                  |label                                    |
    +------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
    |EU rejects German call to boycott British lamb .|[EU, rejects, German, call, to, boycott, British, lamb, .]|[NNP, VBZ, JJ, NN, TO, VB, JJ, NN, .]|[B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]|
    |Peter Blackburn                                 |[Peter, Blackburn]                                        |[NNP, NNP]                           |[B-PER, I-PER]                           |
    |BRUSSELS 1996-08-22                             |[BRUSSELS, 1996-08-22]                                    |[NNP, CD]                            |[B-LOC, O]                               |
    +------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
    """

    def __init__(self,
                 documentCol='document',
                 sentenceCol='sentence',
                 tokenCol='token',
                 posCol='pos',
                 conllLabelIndex=3,
                 conllPosIndex=1,
                 conllDocIdCol="doc_id",
                 textCol='text',
                 labelCol='label',
                 explodeSentences=True,
                 delimiter=' ',
                 includeDocId=False
                 ):
        super(CoNLL, self).__init__("com.johnsnowlabs.nlp.training.CoNLL",
                                    documentCol,
                                    sentenceCol,
                                    tokenCol,
                                    posCol,
                                    conllLabelIndex,
                                    conllPosIndex,
                                    conllDocIdCol,
                                    textCol,
                                    labelCol,
                                    explodeSentences,
                                    delimiter,
                                    includeDocId)

    def readDataset(self, spark, path, read_as=ReadAs.TEXT, partitions=8, storage_level=pyspark.StorageLevel.DISK_ONLY):
        # ToDo Replace with std pyspark
        """Reads the dataset from an external resource.

        Parameters
        ----------
        spark : :class:`pyspark.sql.SparkSession`
            Initiated Spark Session with Spark NLP
        path : str
            Path to the resource, it can take two forms; a path to a conll file, or a path to a folder containing multiple CoNLL files.
            When the path points to a folder, the path must end in '*'.
            Examples:
                "/path/to/single/file.conll'
                "/path/to/folder/containing/multiple/files/*'

        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT
        partitions : sets the minimum number of partitions for the case of lifting multiple files in parallel into a single dataframe. Defaults to 8.
        storage_level : sets the persistence level according to PySpark definitions. Defaults to StorageLevel.DISK_ONLY. Applies only when lifting multiple files.
        

        Returns
        -------
        :class:`pyspark.sql.DataFrame`
            Spark Dataframe with the data
        """
        jSession = spark._jsparkSession

        jdf = self._java_obj.readDataset(jSession, path, read_as, partitions,
                                         spark.sparkContext._getJavaStorageLevel(storage_level))
        dataframe = self.getDataFrame(spark, jdf)
        return dataframe

