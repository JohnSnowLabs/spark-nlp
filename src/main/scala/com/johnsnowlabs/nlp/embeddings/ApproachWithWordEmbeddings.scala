package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorApproach
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}
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
    val currentEmbeddings = {
      clusterEmbeddings
        .orElse(get(sourceEmbeddingsPath).flatMap(sourcePath => {
          EmbeddingsHelper.loadEmbeddings(
            sourcePath,
            spark,
            WordEmbeddingsFormat($(embeddingsFormat)),
            $(embeddingsDim),
            $(caseSensitiveEmbeddings),
            get(includedEmbeddingsRef)
          )
        }))
        .orElse(get(includedEmbeddingsRef).flatMap(EmbeddingsHelper.embeddingsCache.get))
    }

    setEmbeddings(currentEmbeddings
      .getOrElse(throw new IllegalArgumentException(
        s"Word embeddings not found. Either resources not set," +
          s" not found ${get(sourceEmbeddingsPath).getOrElse("")}" +
          s" or not in cache ${get(includedEmbeddingsRef).getOrElse("")}")
      )
    )
  }

  override def onTrained(model: M, spark: SparkSession): Unit = {
    model.setEmbeddings(clusterEmbeddings.get)
    model.setEmbeddingsDim(clusterEmbeddings.get.dim)
    model.setIncludedEmbeddingsIndexPath(clusterEmbeddings.get.clusterFilePath)
    model.setIncludeEmbeddings($(includeEmbeddings))

    get(includedEmbeddingsRef).foreach(ref => model.setIncludedEmbeddingsRef(ref))
  }

  def embeddings: WordEmbeddings = {
    getWordEmbeddings
  }

}


