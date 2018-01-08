package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.ParamsAndFeaturesReadable
import org.apache.spark.sql.SparkSession

trait EmbeddingsReadable[T <: ModelWithWordEmbeddings[_]] extends ParamsAndFeaturesReadable[T] {
  override def onRead(instance: T, path: String, spark: SparkSession): Unit = {
    instance.deserializeEmbeddings(path, spark.sparkContext)
  }
}
