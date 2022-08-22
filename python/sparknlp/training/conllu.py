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
"""Contains classes for CoNLLU."""

from sparknlp.common import ReadAs
from sparknlp.internal import ExtendedJavaWrapper


class CoNLLU(ExtendedJavaWrapper):
    """Instantiates the class to read a CoNLL-U dataset.

    The dataset should be in the format of `CoNLL-U
    <https://universaldependencies.org/format.html>`_ and needs to be specified
    with :meth:`.readDataset`, which will create a dataframe with the data.

    Can be used to train a :class:`DependencyParserApproach
    <sparknlp.annotator.DependencyParserApproach>`

    **Input File Format**::

        # sent_id = 1
        # text = They buy and sell books.
        1   They     they    PRON    PRP    Case=Nom|Number=Plur               2   nsubj   2:nsubj|4:nsubj   _
        2   buy      buy     VERB    VBP    Number=Plur|Person=3|Tense=Pres    0   root    0:root            _
        3   and      and     CONJ    CC     _                                  4   cc      4:cc              _
        4   sell     sell    VERB    VBP    Number=Plur|Person=3|Tense=Pres    2   conj    0:root|2:conj     _
        5   books    book    NOUN    NNS    Number=Plur                        2   obj     2:obj|4:obj       SpaceAfter=No
        6   .        .       PUNCT   .      _                                  2   punct   2:punct           _

    Examples
    --------
    >>> from sparknlp.training import CoNLLU
    >>> conlluFile = "src/test/resources/conllu/en.test.conllu"
    >>> conllDataSet = CoNLLU(False).readDataset(spark, conlluFile)
    >>> conllDataSet.selectExpr(
    ...     "text",
    ...     "form.result as form",
    ...     "upos.result as upos",
    ...     "xpos.result as xpos",
    ...     "lemma.result as lemma"
    ... ).show(1, False)
    +---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
    |text                                   |form                                          |upos                                         |xpos                          |lemma                                       |
    +---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
    |What if Google Morphed Into GoogleOS?  |[What, if, Google, Morphed, Into, GoogleOS, ?]|[PRON, SCONJ, PROPN, VERB, ADP, PROPN, PUNCT]|[WP, IN, NNP, VBD, IN, NNP, .]|[what, if, Google, morph, into, GoogleOS, ?]|
    +---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
    """

    def __init__(self,
                 textCol='text',
                 documentCol='document',
                 sentenceCol='sentence',
                 formCol='form',
                 uposCol='upos',
                 xposCol='xpos',
                 lemmaCol='lemma',
                 explodeSentences=True
                 ):
        super(CoNLLU, self).__init__("com.johnsnowlabs.nlp.training.CoNLLU",
                                     textCol,
                                     documentCol,
                                     sentenceCol,
                                     formCol,
                                     uposCol,
                                     xposCol,
                                     lemmaCol,
                                     explodeSentences)

    def readDataset(self, spark, path, read_as=ReadAs.TEXT):
        """Reads the dataset from an external resource.

        Parameters
        ----------
        spark : :class:`pyspark.sql.SparkSession`
            Initiated Spark Session with Spark NLP
        path : str
            Path to the resource
        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT

        Returns
        -------
        :class:`pyspark.sql.DataFrame`
            Spark Dataframe with the data
        """
        # ToDo Replace with std pyspark
        jSession = spark._jsparkSession

        jdf = self._java_obj.readDataset(jSession, path, read_as)
        dataframe = self.getDataFrame(spark, jdf)
        return dataframe

