package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.ParamsAndFeaturesReadable
import org.apache.spark.sql.SparkSession

trait EmbeddingsReadable[T <: ModelWithWordEmbeddings] extends ParamsAndFeaturesReadable[T] {
  def readEmbeddings(instance: T, path: String, spark: SparkSession): Unit = {
    instance.deserializeEmbeddings(path, spark)
  }

  addReader(readEmbeddings)
}
