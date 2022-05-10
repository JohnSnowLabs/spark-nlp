class Chunk2Doc(AnnotatorTransformer, AnnotatorProperties):
    """Converts a ``CHUNK`` type column back into ``DOCUMENT``. Useful when
    trying to re-tokenize or do further analysis on a ``CHUNK`` result.

    For more extended examples on document pre-processing see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``CHUNK``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    None

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.pretrained import PretrainedPipeline

    Location entities are extracted and converted back into ``DOCUMENT`` type for
    further processing.

    >>> data = spark.createDataFrame([[1, "New York and New Jersey aren't that far apart actually."]]).toDF("id", "text")

    Define pretrained pipeline that extracts Named Entities amongst other things
    and apply `Chunk2Doc` on it.

    >>> pipeline = PretrainedPipeline("explain_document_dl")
    >>> chunkToDoc = Chunk2Doc().setInputCols("entities").setOutputCol("chunkConverted")
    >>> explainResult = pipeline.transform(data)

    Show results.

    >>> result = chunkToDoc.transform(explainResult)
    >>> result.selectExpr("explode(chunkConverted)").show(truncate=False)
    +------------------------------------------------------------------------------+
    |col                                                                           |
    +------------------------------------------------------------------------------+
    |[document, 0, 7, New York, [entity -> LOC, sentence -> 0, chunk -> 0], []]    |
    |[document, 13, 22, New Jersey, [entity -> LOC, sentence -> 0, chunk -> 1], []]|
    +------------------------------------------------------------------------------+

    See Also
    --------
    Doc2Chunk : for converting `DOCUMENT` annotations to `CHUNK`
    """

    name = "Chunk2Doc"

    @keyword_only
    def __init__(self):
        super(Chunk2Doc, self).__init__(classname="com.johnsnowlabs.nlp.Chunk2Doc")

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

