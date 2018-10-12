package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}

trait HasEmbeddings extends AutoCloseable with ParamsAndFeaturesWritable {

  @transient
  protected var clusterEmbeddings: Option[SparkWordEmbeddings] = None

  val includeEmbeddings = new BooleanParam(this, "includeEmbeddings", "whether to include embeddings when saving annotator")
  val includedEmbeddingsRef = new Param[String](this, "includedEmbeddingsRef", "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref")
  val includedEmbeddingsIndexPath = new Param[String](this, "includedEmbeddingsIndexPath", "internal cluster index locator")

  setDefault(includeEmbeddings, true)

  def setIncludeEmbeddings(value: Boolean): this.type = set(this.includeEmbeddings, value)
  def setIncludedEmbeddingsRef(value: String): this.type = set(this.includedEmbeddingsRef, value)
  def setIncludedEmbeddingsIndexPath(value: String): this.type = set(this.includedEmbeddingsIndexPath, value)

  val caseSensitiveEmbeddings = new BooleanParam(this, "caseSensitiveEmbeddings", "whether to ignore case in tokens for embeddings matching")
  val embeddingsDim = new IntParam(this, "embeddingsDim", "Number of embedding dimensions")

  setDefault(caseSensitiveEmbeddings, false)

  def setCaseSensitiveEmbeddings(value: Boolean): this.type = set(this.caseSensitiveEmbeddings, value)
  def setEmbeddingsDim(value: Int): this.type = set(this.embeddingsDim, value)

  def setEmbeddings(embeddings: SparkWordEmbeddings): this.type = {
    set(embeddingsDim, embeddings.dim)
    set(caseSensitiveEmbeddings, embeddings.caseSensitive)
    set(includedEmbeddingsIndexPath, embeddings.clusterFilePath)
    clusterEmbeddings = Some(embeddings)

    this
  }

  def setEmbeddingsIfFNotSet(embeddings: SparkWordEmbeddings): this.type = {
    if (clusterEmbeddings == null || clusterEmbeddings.isEmpty)
      setEmbeddings(embeddings)
    else
      this
  }

  private def updateAvailableEmbeddings(): Unit = {
    /** clusterEmbeddings may become null when a different thread calls getEmbeddings. Clean up now. */
    val cleanEmbeddings: Option[SparkWordEmbeddings] = if (clusterEmbeddings == null) None else clusterEmbeddings
    val currentEmbeddings = cleanEmbeddings
      .orElse(get(includedEmbeddingsRef)
        .flatMap(ref => EmbeddingsHelper.embeddingsCache.get(ref)))
      .orElse(get(includedEmbeddingsIndexPath).filter(_ => $(includeEmbeddings))
        .flatMap(path => EmbeddingsHelper.loadEmbeddings(path, $(embeddingsDim), $(caseSensitiveEmbeddings))))
      .getOrElse(throw new NoSuchElementException(
        s"Word embeddings missing. " +
          s"Not in ref cache ${get(includedEmbeddingsRef).getOrElse("")} " +
          s"or embeddings not included. Check includeEmbeddings or includedEmbeddingsRef params")
      )

    setEmbeddingsIfFNotSet(currentEmbeddings)
  }

  def getEmbeddings: SparkWordEmbeddings = {
    updateAvailableEmbeddings()
    clusterEmbeddings.getOrElse(throw new NoSuchElementException(s"embeddings not set in $uid"))
  }

  def getWordEmbeddings: WordEmbeddings = {
    getEmbeddings.wordEmbeddings
  }

  override def close(): Unit = {
    if (clusterEmbeddings.nonEmpty)
      clusterEmbeddings.get.wordEmbeddings.close()
  }

}
