#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Contains helper classes to make training with specific datasets easier.

To load a specific dataset, the class has to be instantiated, then the data
can be loaded with ``readDataset``.
"""

from sparknlp.internal import ExtendedJavaWrapper
from sparknlp.common import ExternalResource, ReadAs
from pyspark.sql import SparkSession, DataFrame


class CoNLL(ExtendedJavaWrapper):
    def __init__(self,
                 documentCol='document',
                 sentenceCol='sentence',
                 tokenCol='token',
                 posCol='pos',
                 conllLabelIndex=3,
                 conllPosIndex=1,
                 textCol='text',
                 labelCol='label',
                 explodeSentences=True,
                 ):
        """Instantiates the class to read a CoNLL dataset.

        The dataset should be in the format of `CoNLL 2003 <https://www.clips.uantwerpen.be/conll2003/ner/>`_
        and needs to be specified with :meth:`CoNLL.readDataset`.

        Parameters
        ----------
        documentCol : str, optional
            Name of the :class:`DocumentAssembler` column, by default 'document'
        sentenceCol : str, optional
            Name of the :class:`SentenceDetector` column, by default 'sentence'
        tokenCol : str, optional
            Name of the :class:`Tokenizer` column, by default 'token'
        posCol : str, optional
            Name of the :class:`PerceptronApproach` column, by default 'pos'
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
        """
        super(CoNLL, self).__init__("com.johnsnowlabs.nlp.training.CoNLL",
                                    documentCol,
                                    sentenceCol,
                                    tokenCol,
                                    posCol,
                                    conllLabelIndex,
                                    conllPosIndex,
                                    textCol,
                                    labelCol,
                                    explodeSentences)

    def readDataset(self, spark, path, read_as=ReadAs.TEXT):
        # ToDo Replace with std pyspark
        jSession = spark._jsparkSession

        jdf = self._java_obj.readDataset(jSession, path, read_as)
        return DataFrame(jdf, spark._wrapped)


class CoNLLU(ExtendedJavaWrapper):
    def __init__(self, explodeSentences=True):
        """Instantiates the class to read a CoNLL-U dataset.

        The dataset should be in the format of `CoNLL-U <https://universaldependencies.org/format.html>`_
        and needs to be specified with :meth:`CoNLLU.readDataset`.


        Parameters
        ----------
        explodeSentences : bool, optional
            [description], by default True
        """
        super(CoNLLU, self).__init__("com.johnsnowlabs.nlp.training.CoNLLU", explodeSentences)

    def readDataset(self, spark, path, read_as=ReadAs.TEXT):
        # ToDo Replace with std pyspark
        jSession = spark._jsparkSession

        jdf = self._java_obj.readDataset(jSession, path, read_as)
        return DataFrame(jdf, spark._wrapped)


class POS(ExtendedJavaWrapper):
    def __init__(self):
        super(POS, self).__init__("com.johnsnowlabs.nlp.training.POS")

    def readDataset(self, spark, path, delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                    outputTextCol="text"):
        # ToDo Replace with std pyspark
        jSession = spark._jsparkSession

        jdf = self._java_obj.readDataset(jSession, path, delimiter, outputPosCol, outputDocumentCol, outputTextCol)
        return DataFrame(jdf, spark._wrapped)


class PubTator(ExtendedJavaWrapper):
    def __init__(self):
        super(PubTator, self).__init__("com.johnsnowlabs.nlp.training.PubTator")

    def readDataset(self, spark, path, isPaddedToken=True):
        # ToDo Replace with std pyspark
        jSession = spark._jsparkSession

        jdf = self._java_obj.readDataset(jSession, path, isPaddedToken)
        return DataFrame(jdf, spark._wrapped)
