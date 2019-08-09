package com.johnsnowlabs.nlp.embeddings

import org.apache.spark.ml.param.{BooleanParam, Param}

trait HasWordEmbeddings extends HasEmbeddings {

  val embeddingsRef = new Param[String](this, "embeddingsRef", "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref")

  val includeEmbeddings = new BooleanParam(this, "includeEmbeddings", "whether or not to save indexed embeddings along this annotator")

  setDefault(embeddingsRef, this.uid)
  setDefault(includeEmbeddings, true)

  def setEmbeddingsRef(value: String): this.type = {
    if (this.isInstanceOf[WordEmbeddingsModel] && get(embeddingsRef).nonEmpty)
      throw new UnsupportedOperationException(s"Cannot override embeddings ref on a WordEmbeddingsModel. Please re-use current ref: $getEmbeddingsRef")
    set(this.embeddingsRef, value)
  }
  def getEmbeddingsRef: String = $(embeddingsRef)

  def setIncludeEmbeddings(value: Boolean): this.type = set(includeEmbeddings, value)
  def getIncludeEmbeddings: Boolean = $(includeEmbeddings)

  @transient private var wembeddings: WordEmbeddingsRetriever = null
  @transient private var loaded: Boolean = false

  protected def setAsLoaded(): Unit = loaded = true
  protected def isLoaded(): Boolean = loaded

  protected def getEmbeddings: WordEmbeddingsRetriever = {
    if (Option(wembeddings).isDefined)
      wembeddings
    else {
      wembeddings = getClusterEmbeddings.getLocalRetriever
      wembeddings
    }
  }

  protected var preloadedEmbeddings: Option[ClusterWordEmbeddings] = None

  protected def getClusterEmbeddings: ClusterWordEmbeddings = {
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

}
