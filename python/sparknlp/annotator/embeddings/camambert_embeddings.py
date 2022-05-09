class CamemBertEmbeddings(AnnotatorModel,
                          HasEmbeddingsProperties,
                          HasCaseSensitiveProperties,
                          HasStorageRef,
                          HasBatchedAnnotate):
    name = "CamemBertEmbeddings"

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.CamemBertEmbeddings", java_model=None):
        super(CamemBertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            dimension=768,
            maxSentenceLength=128,
            caseSensitive=True
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        CamemBertEmbeddings
            The restored model
        """
        from sparknlp.internal import _CamemBertLoader
        jModel = _CamemBertLoader(folder, spark_session._jsparkSession)._java_obj
        return CamemBertEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="camembert_base", lang="fr", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "albert_base_uncased"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        CamemBertEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(CamemBertEmbeddings, name, lang, remote_loc)
