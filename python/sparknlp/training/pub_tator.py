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
