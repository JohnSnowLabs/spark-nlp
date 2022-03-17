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

"""Contains helper classes to make training with specific datasets easier.

To load a specific dataset, the class has to be instantiated, then the data
can be loaded with ``readDataset``.
"""

from sparknlp.internal import ExtendedJavaWrapper
from sparknlp.common import ExternalResource, ReadAs
from pyspark.sql import SparkSession, DataFrame
import pyspark


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
                 textCol='text',
                 labelCol='label',
                 explodeSentences=True,
                 delimiter=' '
                 ):
        super(CoNLL, self).__init__("com.johnsnowlabs.nlp.training.CoNLL",
                                    documentCol,
                                    sentenceCol,
                                    tokenCol,
                                    posCol,
                                    conllLabelIndex,
                                    conllPosIndex,
                                    textCol,
                                    labelCol,
                                    explodeSentences,
                                    delimiter)

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
        return DataFrame(jdf, spark._wrapped)


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
        return DataFrame(jdf, spark._wrapped)


class POS(ExtendedJavaWrapper):
    """Helper class for creating DataFrames for training a part-of-speech
    tagger.

    The dataset needs to consist of sentences on each line, where each word is
    delimited with its respective tag.

    **Input File Format**::

        A|DT few|JJ months|NNS ago|RB you|PRP received|VBD a|DT letter|NN


    The sentence can then be parsed with :meth:`.readDataset` into a column with
    annotations of type ``POS``.

    Can be used to train a :class:`PerceptronApproach
    <sparknlp.annotator.PerceptronApproach>`.

    Examples
    --------
    In this example, the file ``test-training.txt`` has the content of the
    sentence above.

    >>> from sparknlp.training import POS
    >>> pos = POS()
    >>> path = "src/test/resources/anc-pos-corpus-small/test-training.txt"
    >>> posDf = pos.readDataset(spark, path, "|", "tags")
    >>> posDf.selectExpr("explode(tags) as tags").show(truncate=False)
    +---------------------------------------------+
    |tags                                         |
    +---------------------------------------------+
    |[pos, 0, 5, NNP, [word -> Pierre], []]       |
    |[pos, 7, 12, NNP, [word -> Vinken], []]      |
    |[pos, 14, 14, ,, [word -> ,], []]            |
    |[pos, 16, 17, CD, [word -> 61], []]          |
    |[pos, 19, 23, NNS, [word -> years], []]      |
    |[pos, 25, 27, JJ, [word -> old], []]         |
    |[pos, 29, 29, ,, [word -> ,], []]            |
    |[pos, 31, 34, MD, [word -> will], []]        |
    |[pos, 36, 39, VB, [word -> join], []]        |
    |[pos, 41, 43, DT, [word -> the], []]         |
    |[pos, 45, 49, NN, [word -> board], []]       |
    |[pos, 51, 52, IN, [word -> as], []]          |
    |[pos, 47, 47, DT, [word -> a], []]           |
    |[pos, 56, 67, JJ, [word -> nonexecutive], []]|
    |[pos, 69, 76, NN, [word -> director], []]    |
    |[pos, 78, 81, NNP, [word -> Nov.], []]       |
    |[pos, 83, 84, CD, [word -> 29], []]          |
    |[pos, 81, 81, ., [word -> .], []]            |
    +---------------------------------------------+
    """

    def __init__(self):
        super(POS, self).__init__("com.johnsnowlabs.nlp.training.POS")

    def readDataset(self, spark, path, delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                    outputTextCol="text"):
        # ToDo Replace with std pyspark
        """Reads the dataset from an external resource.

        Parameters
        ----------
        spark : :class:`pyspark.sql.SparkSession`
            Initiated Spark Session with Spark NLP
        path : str
            Path to the resource
        delimiter : str, optional
            Delimiter of word and POS, by default "|"
        outputPosCol : str, optional
            Name of the output POS column, by default "tags"
        outputDocumentCol : str, optional
            Name of the output document column, by default "document"
        outputTextCol : str, optional
            Name of the output text column, by default "text"

        Returns
        -------
        :class:`pyspark.sql.DataFrame`
            Spark Dataframe with the data
        """
        jSession = spark._jsparkSession

        jdf = self._java_obj.readDataset(jSession, path, delimiter, outputPosCol, outputDocumentCol, outputTextCol)
        return DataFrame(jdf, spark._wrapped)


class PubTator(ExtendedJavaWrapper):
    """The PubTator format includes medical papers’ titles, abstracts, and
    tagged chunks.

    For more information see `PubTator Docs
    <http://bioportal.bioontology.org/ontologies/EDAM?p=classes&conceptid=format_3783>`_
    and `MedMentions Docs <http://github.com/chanzuckerberg/MedMentions>`_.

    :meth:`.readDataset` is used to create a Spark DataFrame from a PubTator
    text file.

    **Input File Format**::

        25763772	0	5	DCTN4	T116,T123	C4308010
        25763772	23	63	chronic Pseudomonas aeruginosa infection	T047	C0854135
        25763772	67	82	cystic fibrosis	T047	C0010674
        25763772	83	120	Pseudomonas aeruginosa (Pa) infection	T047	C0854135
        25763772	124	139	cystic fibrosis	T047	C0010674

    Examples
    --------
    >>> from sparknlp.training import PubTator
    >>> pubTatorFile = "./src/test/resources/corpus_pubtator_sample.txt"
    >>> pubTatorDataSet = PubTator().readDataset(spark, pubTatorFile)
    >>> pubTatorDataSet.show(1)
    +--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
    |  doc_id|      finished_token|        finished_pos|        finished_ner|finished_token_metadata|finished_pos_metadata|finished_label_metadata|
    +--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
    |25763772|[DCTN4, as, a, mo...|[NNP, IN, DT, NN,...|[B-T116, O, O, O,...|   [[sentence, 0], [...| [[word, DCTN4], [...|   [[word, DCTN4], [...|
    +--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
    """

    def __init__(self):
        super(PubTator, self).__init__("com.johnsnowlabs.nlp.training.PubTator")

    def readDataset(self, spark, path, isPaddedToken=True):
        # ToDo Replace with std pyspark
        """Reads the dataset from an external resource.

        Parameters
        ----------
        spark : :class:`pyspark.sql.SparkSession`
            Initiated Spark Session with Spark NLP
        path : str
            Path to the resource
        isPaddedToken : str, optional
            Whether tokens are padded, by default True

        Returns
        -------
        :class:`pyspark.sql.DataFrame`
            Spark Dataframe with the data
        """
        jSession = spark._jsparkSession

        jdf = self._java_obj.readDataset(jSession, path, isPaddedToken)
        return DataFrame(jdf, spark._wrapped)
