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
"""Contains base classes for NER Annotators."""

from sparknlp.common import *


class NerApproach(Params):
    """Base class for Ner*Approach Annotators
    """
    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Column with label per each token",
                        typeConverter=TypeConverters.toString)

    entities = Param(Params._dummy(), "entities", "Entities to recognize", TypeConverters.toListString)

    minEpochs = Param(Params._dummy(), "minEpochs", "Minimum number of epochs to train", TypeConverters.toInt)

    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)

    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)

    def setLabelColumn(self, value):
        """Sets name of column for data labels.

        Parameters
        ----------
        value : str
            Column for data labels
        """
        return self._set(labelColumn=value)

    def setEntities(self, tags):
        """Sets entities to recognize.

        Parameters
        ----------
        tags : List[str]
            List of entities
        """
        return self._set(entities=tags)

    def setMinEpochs(self, epochs):
        """Sets minimum number of epochs to train.

        Parameters
        ----------
        epochs : int
            Minimum number of epochs to train
        """
        return self._set(minEpochs=epochs)

    def setMaxEpochs(self, epochs):
        """Sets maximum number of epochs to train.

        Parameters
        ----------
        epochs : int
            Maximum number of epochs to train
        """
        return self._set(maxEpochs=epochs)

    def setRandomSeed(self, seed):
        """Sets random seed for shuffling.

        Parameters
        ----------
        seed : int
            Random seed for shuffling
        """
        return self._set(randomSeed=seed)

    def getLabelColumn(self):
        """Gets column for label per each token.

        Returns
        -------
        str
            Column with label per each token
        """
        return self.getOrDefault(self.labelColumn)

