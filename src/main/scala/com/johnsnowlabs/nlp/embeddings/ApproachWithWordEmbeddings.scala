package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorApproach
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.sql.SparkSession


/**
  * Base class for annotators that uses Word Embeddings.
  * This implementation is based on RocksDB so it has a compact RAM usage
  *
  * 1. User configures Word Embeddings by method 'setWordEmbeddingsSource'.
  * 2. During training Word Embeddings are indexed as RockDB index file.
  * 3. Than this index file is spread across the cluster.
  * 4. Every model 'ModelWithWordEmbeddings' uses local RocksDB as Word Embeddings lookup.
 */

// had to relax the requirement for type M here - check.
abstract class ApproachWithWordEmbeddings[A <: ApproachWithWordEmbeddings[A, M], M <: Model[M] with ModelWithWordEmbeddings]
  extends AnnotatorApproach[M] with HasEmbeddings {

  val sourceEmbeddingsPath = new Param[String](this, "sourceEmbeddingsPath", "Word embeddings file")
  val embeddingsFormat = new IntParam(this, "embeddingsFormat", "Word vectors file format")

  def setEmbeddingsSource(path: String, nDims: Int, format: WordEmbeddingsFormat.Format): A = {
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
    set(this.embeddingsDim, nDims).asInstanceOf[A]
  }

  def setEmbeddingsSource(path: String, nDims: Int, format: String): A = {
    import WordEmbeddingsFormat._
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
    set(this.embeddingsDim, nDims).asInstanceOf[A]
  }

  override def beforeTraining(spark: SparkSession): Unit = {
    val clusterEmbeddings = {
      if (isDefined(sourceEmbeddingsPath)) {
        EmbeddingsHelper.loadEmbeddings(
          $(sourceEmbeddingsPath),
          spark,
          WordEmbeddingsFormat($(embeddingsFormat)).toString,
          $(embeddingsDim),
          $(caseSensitiveEmbeddings)
        )
      } else if (isSet(embeddingsRef)) {
        EmbeddingsHelper.getEmbeddingsByRef($(embeddingsRef))
          .map(clusterEmbeddings => {
            set(embeddingsDim, clusterEmbeddings.dim)
            set(caseSensitiveEmbeddings, clusterEmbeddings.caseSensitive)
            clusterEmbeddings
          }).getOrElse(throw new NoSuchElementException(s"embeddings by ref ${$(embeddingsRef)} not found"))
      } else
        throw new IllegalArgumentException(
          s"Word embeddings not found. Either sourceEmbeddingsPath not set," +
            s" or not in cache by ref: ${get(embeddingsRef).getOrElse("-embeddingsRef not set-")}. " +
            s"Load using EmbeddingsHelper .loadEmbeddings() and .setEmbeddingsRef() to make them available."
        )
    }

    /** Set embeddings ref */
    EmbeddingsHelper.setEmbeddingsRef($(embeddingsRef), clusterEmbeddings)

  }

  override def onTrained(model: M, spark: SparkSession): Unit = {
    val clusterEmbeddings = EmbeddingsHelper.getEmbeddingsByRef($(embeddingsRef))
      .getOrElse(throw new NoSuchElementException("Embeddings not found after training"))

    model.setIncludeEmbeddings($(includeEmbeddings))
    model.setEmbeddingsDim(clusterEmbeddings.dim)
    model.setCaseSensitiveEmbeddings(clusterEmbeddings.caseSensitive)

    if (isSet(embeddingsRef)) model.setEmbeddingsRef($(embeddingsRef))

  }

}


