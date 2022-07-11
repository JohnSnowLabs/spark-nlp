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
"""Contains helper classes for part-of-speech tagging."""

from sparknlp.internal import ExtendedJavaWrapper


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
        dataframe = self.getDataFrame(spark, jdf)
        return dataframe
