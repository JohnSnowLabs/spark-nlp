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
