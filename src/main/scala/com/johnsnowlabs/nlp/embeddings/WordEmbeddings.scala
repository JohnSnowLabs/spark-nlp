package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesWritable}
import com.johnsnowlabs.nlp.annotators.common.{TokenPieceEmbeddings, TokenizedWithSentence, WordpieceEmbeddingsSentence}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}


class WordEmbeddings(override val uid: String)
  extends AnnotatorModel[WordEmbeddings]
    with HasEmbeddings
    with AutoCloseable
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("EMBEDDINGS_LOOKUP_MODEL"))

  val sourceEmbeddingsPath = new Param[String](this, "sourceEmbeddingsPath", "Word embeddings file")
  val embeddingsFormat = new IntParam(this, "embeddingsFormat", "Word vectors file format")

  val embeddingsRef = new Param[String](this, "embeddingsRef", "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref")

  setDefault(embeddingsRef, this.uid)

  def setEmbeddingsRef(value: String): this.type = set(this.embeddingsRef, value)
  def getEmbeddingsRef: String = $(embeddingsRef)

  private def getEmbeddingsSerializedPath(path: String): Path =
    Path.mergePaths(new Path(path), new Path("/embeddings"))

  private[embeddings] def deserializeEmbeddings(path: String, spark: SparkSession): Unit = {
    val src = getEmbeddingsSerializedPath(path)

    EmbeddingsHelper.load(
      src.toUri.toString,
      spark,
      WordEmbeddingsFormat.SPARKNLP.toString,
      $(dimension),
      $(caseSensitive),
      $(embeddingsRef)
    )
  }

  private[embeddings] def serializeEmbeddings(path: String, spark: SparkSession): Unit = {
    val index = new Path(EmbeddingsHelper.getLocalEmbeddingsPath(getClusterEmbeddings.fileName))

    val uri = new java.net.URI(path)
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = getEmbeddingsSerializedPath(path)

    EmbeddingsHelper.save(fs, index, dst)
  }

  override protected def onWrite(path: String, spark: SparkSession): Unit = {
    /** Param only useful for runtime execution */
    serializeEmbeddings(path, spark)
  }

  @transient private var wembeddings: WordEmbeddingsRetriever = null

  private def getEmbeddings: WordEmbeddingsRetriever = {
    if (Option(wembeddings).isDefined)
      wembeddings
    else {
      wembeddings = getClusterEmbeddings.getLocalRetriever
      wembeddings
    }
  }

  private var preloadedEmbeddings: Option[ClusterWordEmbeddings] = None

  private[embeddings] def getClusterEmbeddings: ClusterWordEmbeddings = {
    if (preloadedEmbeddings.isDefined && preloadedEmbeddings.get.fileName == $(embeddingsRef))
      return preloadedEmbeddings.get
    else {
      preloadedEmbeddings.foreach(_.getLocalRetriever.close())
      preloadedEmbeddings = Some(EmbeddingsHelper.load(
        EmbeddingsHelper.getClusterFilename($(embeddingsRef)),
        $(dimension),
        $(caseSensitive)
      ))
    }
    preloadedEmbeddings.get
  }

  override protected def close(): Unit = {
    get(embeddingsRef)
      .flatMap(_ => preloadedEmbeddings)
      .foreach(_.getLocalRetriever.close())
  }

  def setEmbeddingsSource(path: String, nDims: Int, format: WordEmbeddingsFormat.Format): this.type = {
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
    set(this.dimension, nDims)
  }

  def setEmbeddingsSource(path: String, nDims: Int, format: String): this.type = {
    import WordEmbeddingsFormat._
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
    set(this.dimension, nDims)
  }

  override protected def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    if (isDefined(sourceEmbeddingsPath)) {
      EmbeddingsHelper.load(
        $(sourceEmbeddingsPath),
        dataset.sparkSession,
        WordEmbeddingsFormat($(embeddingsFormat)).toString,
        $(dimension),
        $(caseSensitive),
        $(embeddingsRef)
      )
    } else if (isSet(embeddingsRef)) {
      getClusterEmbeddings
    } else
      throw new IllegalArgumentException(
        s"Word embeddings not found. Either sourceEmbeddingsPath not set," +
          s" or not in cache by ref: ${get(embeddingsRef).getOrElse("-embeddingsRef not set-")}. " +
          s"Load using EmbeddingsHelper .loadEmbeddings() and .setEmbeddingsRef() to make them available."
      )
    dataset
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {

    setDimension($(dimension))
    setCaseSensitive($(caseSensitive))

    if (isSet(embeddingsRef)) setEmbeddingsRef($(embeddingsRef))

    getClusterEmbeddings.getLocalRetriever.close()

    dataset.withColumn(getOutputCol, wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension)))

  }

  /**
    * takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = TokenizedWithSentence.unpack(annotations)
    val withEmbeddings = sentences.zipWithIndex.map{case (s, idx) =>
      val tokens = s.indexedTokens.map {token =>
        val vector = this.getEmbeddings.getEmbeddingsVector(token.token)
        new TokenPieceEmbeddings(token.token, token.token, -1, true, vector, token.begin, token.end)
      }
      WordpieceEmbeddingsSentence(tokens, idx)
    }

    WordpieceEmbeddingsSentence.pack(withEmbeddings)
  }

  override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)
}

object WordEmbeddings extends EmbeddingsReadable[WordEmbeddings]
