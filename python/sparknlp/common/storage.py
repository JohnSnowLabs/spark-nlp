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
"""Contains utility classes for handling storage."""

from pyspark.ml.param import Param, Params, TypeConverters

from sparknlp.common.utils import ExternalResource
from sparknlp.common.properties import HasCaseSensitiveProperties
import sparknlp.internal as _internal


class HasStorageRef:
    storageRef = Param(Params._dummy(), "storageRef",
                       "unique reference name for identification",
                       TypeConverters.toString)

    def setStorageRef(self, value):
        """Sets unique reference name for identification.

        Parameters
        ----------
        value : str
            Unique reference name for identification
        """
        return self._set(storageRef=value)

    def getStorageRef(self):
        """Gets unique reference name for identification.

        Returns
        -------
        str
            Unique reference name for identification
        """
        return self.getOrDefault("storageRef")


class HasStorageOptions:
    includeStorage = Param(Params._dummy(),
                           "includeStorage",
                           "whether to include indexed storage in trained model",
                           typeConverter=TypeConverters.toBoolean)

    enableInMemoryStorage = Param(Params._dummy(),
                                  "enableInMemoryStorage",
                                  "whether to load whole indexed storage in memory (in-memory lookup)",
                                  typeConverter=TypeConverters.toBoolean)

    def setIncludeStorage(self, value):
        """Sets whether to include indexed storage in trained model.

        Parameters
        ----------
        value : bool
            Whether to include indexed storage in trained model
        """
        return self._set(includeStorage=value)

    def getIncludeStorage(self):
        """Gets whether to include indexed storage in trained model.

        Returns
        -------
        bool
            Whether to include indexed storage in trained model
        """
        return self.getOrDefault("includeStorage")

    def setEnableInMemoryStorage(self, value):
        """Sets whether to load whole indexed storage in memory (in-memory lookup)

        Parameters
        ----------
        value : bool
            Whether to load whole indexed storage in memory (in-memory lookup)
        """
        return self._set(enableInMemoryStorage=value)

    def getEnableInMemoryStorage(self):
        return self.getOrDefault("enableInMemoryStorage")


class HasStorageModel(HasStorageRef, HasCaseSensitiveProperties, HasStorageOptions):

    def saveStorage(self, path, spark):
        """Saves the current model to storage.

        Parameters
        ----------
        path : str
            Path for saving the model.
        spark : :class:`pyspark.sql.SparkSession`
            The current SparkSession
        """
        self._transfer_params_to_java()
        self._java_obj.saveStorage(path, spark._jsparkSession, False)

    @staticmethod
    def loadStorage(path, spark, storage_ref):
        raise NotImplementedError("AnnotatorModel with HasStorageModel did not implement 'loadStorage'")

    @staticmethod
    def loadStorages(path, spark, storage_ref, databases):
        for database in databases:
            _internal._StorageHelper(path, spark, database, storage_ref, within_storage=False)


class HasStorage(HasStorageRef, HasCaseSensitiveProperties, HasStorageOptions):
    storagePath = Param(Params._dummy(),
                        "storagePath",
                        "path to file",
                        typeConverter=TypeConverters.identity)

    def setStoragePath(self, path, read_as):
        """Sets path to file.

        Parameters
        ----------
        path : str
            Path to file
        read_as : str
            How to interpret the file

        Notes
        -----
        See :class:`ReadAs <sparknlp.common.ReadAs>` for reading options.
        """
        return self._set(storagePath=ExternalResource(path, read_as, {}))

    def getStoragePath(self):
        """Gets path to file.

        Returns
        -------
        str
            path to file
        """
        return self.getOrDefault("storagePath")
