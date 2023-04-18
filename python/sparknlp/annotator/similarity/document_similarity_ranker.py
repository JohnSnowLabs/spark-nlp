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
"""Contains classes for DocumentSimilarityRanker."""

from sparknlp.common import *


class DocumentSimilarityRankerApproach(AnnotatorApproach, HasEnableCachingProperties):
    inputAnnotatorTypes = [AnnotatorType.SENTENCE_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.DOC_SIMILARITY_RANKINGS

    similarityMethod = Param(Params._dummy(),
                             "similarityMethod",
                             "The similarity method used to calculate the neighbours. (Default: 'brp', "
                             "Bucketed Random Projection for Euclidean Distance)",
                             typeConverter=TypeConverters.toString)

    numberOfNeighbours = Param(Params._dummy(),
                               "numberOfNeighbours",
                               "The number of neighbours the model will return (Default:`10`)",
                               typeConverter=TypeConverters.toInt)

    bucketLength = Param(Params._dummy(),
                         "bucketLength",
                         "The bucket length that controls the average size of hash buckets. "
                         "A larger bucket length (i.e., fewer buckets) increases the probability of features "
                         "being hashed to the same bucket (increasing the numbers of true and false positives).",
                         typeConverter=TypeConverters.toFloat)

    numHashTables = Param(Params._dummy(),
                          "numHashTables",
                          "number of hash tables, where increasing number of hash tables lowers the "
                          "false negative rate,and decreasing it improves the running performance.",
                          typeConverter=TypeConverters.toInt)

    visibleDistances = Param(Params._dummy(),
                             "visibleDistances",
                             "Whether to set visibleDistances in ranking output (Default: `false`).",
                             typeConverter=TypeConverters.toBoolean)

    identityRanking = Param(Params._dummy(),
                            "identityRanking",
                            "Whether to include identity in ranking result set. Useful for debug. (Default: `false`).",
                            typeConverter=TypeConverters.toBoolean)

    def setSimilarityMethod(self, value):
        """Sets the similarity method used to calculate the neighbours.
            (Default: `"brp"`, Bucketed Random Projection for Euclidean Distance)

        Parameters
        ----------
        value : str
            the similarity method to calculate the neighbours.
        """
        return self._set(similarityMethod=value)

    def setNumberOfNeighbours(self, value):
        """Sets The number of neighbours the model will return for each document(Default:`"10"`).

        Parameters
        ----------
        value : str
            the number of neighbours the model will return for each document.
        """
        return self._set(numberOfNeighbours=value)

    def setBucketLength(self, value):
        """Sets the bucket length that controls the average size of hash buckets (Default:`"2.0"`).

        Parameters
        ----------
        value : float
            Sets the bucket length that controls the average size of hash buckets.
        """
        return self._set(bucketLength=value)

    def setNumHashTables(self, value):
        """Sets the number of hash tables.

        Parameters
        ----------
        value : int
            Sets the number of hash tables.
        """
        return self._set(numHashTables=value)

    def setVisibleDistances(self, value):
        """Sets the document distances visible in the result set.

        Parameters
        ----------
        value : bool
            Sets the document distances visible in the result set.
            Default('False')
        """
        return self._set(visibleDistances=value)

    def setIdentityRanking(self, value):
        """Sets the document identity ranking inclusive in the result set.

        Parameters
        ----------
        value : bool
            Sets the document identity ranking inclusive in the result set.
            Useful for debugging.
            Default('False').
        """
        return self._set(identityRanking=value)

    @keyword_only
    def __init__(self):
        super(DocumentSimilarityRankerApproach, self)\
            .__init__(classname="com.johnsnowlabs.nlp.annotators.similarity.DocumentSimilarityRankerApproach")
        self._setDefault(
            similarityMethod="brp",
            numberOfNeighbours=10,
            bucketLength=2.0,
            numHashTables=3,
            visibleDistances=False,
            identityRanking=False
        )

    def _create_model(self, java_model):
        return DocumentSimilarityRankerModel(java_model=java_model)


class DocumentSimilarityRankerModel(AnnotatorModel, HasEmbeddingsProperties):

    name = "DocumentSimilarityRankerModel"
    inputAnnotatorTypes = [AnnotatorType.SENTENCE_EMBEDDINGS]
    outputAnnotatorType = AnnotatorType.DOC_SIMILARITY_RANKINGS

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.similarity.DocumentSimilarityRankerModel",
                 java_model=None):
        super(DocumentSimilarityRankerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )


class DocumentSimilarityRankerFinisher:
    pass
