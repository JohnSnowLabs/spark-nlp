package com.johnsnowlabs.nlp.embeddings

import org.apache.spark.ml.param.Param

trait HasWordEmbeddings extends HasEmbeddings {

  val embeddingsRef = new Param[String](this, "embeddingsRef", "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref")

  setDefault(embeddingsRef, this.uid)

  def setEmbeddingsRef(value: String): this.type = set(this.embeddingsRef, value)
  def getEmbeddingsRef: String = $(embeddingsRef)

  @transient private var wembeddings: WordEmbeddingsRetriever = null

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
