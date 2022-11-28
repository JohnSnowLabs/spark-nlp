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
from sparknlp.internal import ParamsGettersSetters


class EvaluationDLParams(ParamsGettersSetters):

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)

    validationSplit = Param(Params._dummy(), "validationSplit",
                            "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    evaluationLogExtended = Param(Params._dummy(), "evaluationLogExtended",
                                  "Whether logs for validation to be extended: it displays time and evaluation of each label. Default is False.",
                                  TypeConverters.toBoolean)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

    outputLogsPath = Param(Params._dummy(), "outputLogsPath", "Folder path to save training logs",
                           TypeConverters.toString)

    testDataset = Param(Params._dummy(), "testDataset",
                        "Path to test dataset. If set used to calculate statistic on it during training.",
                        TypeConverters.identity)

    def setVerbose(self, value):
        """Sets level of verbosity during training

        Parameters
        ----------
        value : int
            Level of verbosity
        """
        return self._set(verbose=value)

    def setValidationSplit(self, v):
        """Sets the proportion of training dataset to be validated against the
        model on each Epoch, by default it is 0.0 and off. The value should be
        between 0.0 and 1.0.

        Parameters
        ----------
        v : float
            Proportion of training dataset to be validated
        """
        self._set(validationSplit=v)
        return self

    def setEvaluationLogExtended(self, v):
        """Sets whether logs for validation to be extended, by default False.
        Displays time and evaluation of each label.

        Parameters
        ----------
        v : bool
            Whether logs for validation to be extended

        """
        self._set(evaluationLogExtended=v)
        return self

    def setEnableOutputLogs(self, value):
        """Sets whether to use stdout in addition to Spark logs, by default
        False.

        Parameters
        ----------
        value : bool
            Whether to use stdout in addition to Spark logs
        """
        return self._set(enableOutputLogs=value)

    def setOutputLogsPath(self, p):
        """Sets folder path to save training logs

        Parameters
        ----------
        p : str
            Folder path to save training logs
        """
        return self._set(outputLogsPath=p)

    def setTestDataset(self, path, read_as=ReadAs.SPARK, options={"format": "parquet"}):
        """Path to a parquet file of a test dataset. If set, it is used to calculate
        statistics on it during training.

        The parquet file must be a dataframe that has the same columns as the model that
        is being trained. For example, if the model needs as input `DOCUMENT`, `TOKEN`,
        `WORD_EMBEDDINGS` (Features) and `NAMED_ENTITY` (label) then these columns also
        need to be present while saving the dataframe. The pre-processing steps for the
        training dataframe should also be applied to the test dataframe.

        An example on how to create such a parquet file could be:

        >>> # assuming preProcessingPipeline
        >>> (train, test) = data.randomSplit([0.8, 0.2])
        >>> preProcessingPipeline
        ...     .fit(test)
        ...     .transform(test)
        ...     .write
        ...     .mode("overwrite")
        ...     .parquet("test_data")
        >>> annotator.setTestDataset("test_data")

        Parameters
        ----------
        path : str
            Path to test dataset
        read_as : str, optional
            How to read the resource, by default ReadAs.SPARK
        options : dict, optional
            Options for reading the resource, by default {"format": "csv"}
        """
        return self._set(testDataset=ExternalResource(path, read_as, options.copy()))
