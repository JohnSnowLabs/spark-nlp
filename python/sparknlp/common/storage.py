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

class HasExcludableStorage:
    includeStorage = Param(Params._dummy(),
                           "includeStorage",
                           "whether to include indexed storage in trained model",
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

class HasStorageModel(HasStorageRef, HasCaseSensitiveProperties, HasExcludableStorage):

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

class HasStorage(HasStorageRef, HasCaseSensitiveProperties, HasExcludableStorage):
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

