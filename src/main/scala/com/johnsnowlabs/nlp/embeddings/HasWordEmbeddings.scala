package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.HasEmbeddingsRef
import org.apache.spark.ml.param.BooleanParam

trait HasWordEmbeddings extends EmbeddingsProperties with HasEmbeddingsRef {

  val includeEmbeddings = new BooleanParam(this, "includeEmbeddings", "whether or not to save indexed embeddings along this annotator")

  setDefault(includeEmbeddings, true)

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
