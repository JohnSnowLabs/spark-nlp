package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}

trait HasEmbeddings extends AutoCloseable with ParamsAndFeaturesWritable {

  val embeddingsRef = new Param[String](this, "embeddingsRef", "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref")
  val embeddingsDim = new IntParam(this, "embeddingsDim", "Number of embedding dimensions")
  val caseSensitiveEmbeddings = new BooleanParam(this, "caseSensitiveEmbeddings", "whether to ignore case in tokens for embeddings matching")

  setDefault(caseSensitiveEmbeddings, false)
  setDefault(embeddingsRef, this.uid)

  def setEmbeddingsRef(value: String): this.type = set(this.embeddingsRef, value)
  def setEmbeddingsDim(value: Int): this.type = set(this.embeddingsDim, value)
  def setCaseSensitiveEmbeddings(value: Boolean): this.type = set(this.caseSensitiveEmbeddings, value)

  def getEmbeddingsRef: String = $(embeddingsRef)
  def getEmbeddingsDim: Int = $(embeddingsDim)
  def getCaseSensitiveEmbeddings: Boolean = $(caseSensitiveEmbeddings)

  @transient private var wembeddings: WordEmbeddingsRetriever = null

  def getEmbeddings: WordEmbeddingsRetriever = {
    if (Option(wembeddings).isDefined)
      wembeddings
    else {
      wembeddings = getClusterEmbeddings.getLocalRetriever
      wembeddings
    }
  }

  private var preloadedEmbeddings: Option[ClusterWordEmbeddings] = None

  def getClusterEmbeddings: ClusterWordEmbeddings = {
    if (preloadedEmbeddings.isDefined && preloadedEmbeddings.get.fileName == $(embeddingsRef))
      return preloadedEmbeddings.get
    else {
      preloadedEmbeddings.foreach(_.getLocalRetriever.close())
      preloadedEmbeddings = Some(EmbeddingsHelper.load(
        EmbeddingsHelper.getClusterFilename($(embeddingsRef)),
        $(embeddingsDim),
        $(caseSensitiveEmbeddings)
      ))
    }
    preloadedEmbeddings.get

  }

  override def close(): Unit = {
    get(embeddingsRef)
      .flatMap(_ => preloadedEmbeddings)
      .foreach(_.getLocalRetriever.close())
  }

}
