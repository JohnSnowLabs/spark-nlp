package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.{AnnotatorApproach, HasWordEmbeddings}
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
abstract class ApproachWithWordEmbeddings[A <: ApproachWithWordEmbeddings[A, M], M <: Model[M] with HasWordEmbeddings]
  extends AnnotatorApproach[M] with AutoCloseable {

  val sourceEmbeddingsPath = new Param[String](this, "sourceEmbeddingsPath", "Word embeddings file")
  val embeddingsFormat = new IntParam(this, "embeddingsFormat", "Word vectors file format")
  val embeddingsNDims = new IntParam(this, "embeddingsNDims", "Number of dimensions for word vectors")
  val normalizeEmbeddings = new BooleanParam(this, "normalizeEmbeddings", "whether to use embeddings of normalized tokens (if not already normalized)")

  setDefault(normalizeEmbeddings, true)

  def setEmbeddingsSource(path: String, nDims: Int, format: WordEmbeddingsFormat.Format): A = {
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
    set(this.embeddingsNDims, nDims).asInstanceOf[A]
  }

  def setEmbeddingsSource(path: String, nDims: Int, format: String): A = {
    import WordEmbeddingsFormat._
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
    set(this.embeddingsNDims, nDims).asInstanceOf[A]
  }

  override def beforeTraining(spark: SparkSession): Unit = {
    if (isDefined(sourceEmbeddingsPath)) {
      clusterEmbeddings = Some(SparkWordEmbeddings(
        spark.sparkContext,
        $(sourceEmbeddingsPath),
        $(embeddingsNDims),
        $(normalizeEmbeddings),
        WordEmbeddingsFormat($(embeddingsFormat))
      ))
    }
  }


  override def onTrained(model: M, spark: SparkSession): Unit = {
    if (isDefined(sourceEmbeddingsPath)) {
      model.setDims($(embeddingsNDims))
      model.setIndexPath(clusterEmbeddings.get.clusterFilePath.toString)
    }
  }

  private var clusterEmbeddings: Option[SparkWordEmbeddings] = None

  def embeddings: Option[WordEmbeddings] = {
    clusterEmbeddings.map(c => c.wordEmbeddings)
  }

  override def close(): Unit = {
    if (embeddings.nonEmpty)
      embeddings.get.close()
  }
}


