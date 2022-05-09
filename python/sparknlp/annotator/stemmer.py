class Stemmer(AnnotatorModel):
    """Returns hard-stems out of words with the objective of retrieving the
    meaningful part of the word.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    None

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> stemmer = Stemmer() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("stem")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     stemmer
    ... ])
    >>> data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers."]]) \\
    ...     .toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("stem.result").show(truncate = False)
    +-------------------------------------------------------------+
    |result                                                       |
    +-------------------------------------------------------------+
    |[peter, piper, employe, ar, pick, peck, of, pickl, pepper, .]|
    +-------------------------------------------------------------+
    """

    language = Param(Params._dummy(), "language", "stemmer algorithm", typeConverter=TypeConverters.toString)

    name = "Stemmer"

    @keyword_only
    def __init__(self):
        super(Stemmer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Stemmer")
        self._setDefault(
            language="english"
        )

