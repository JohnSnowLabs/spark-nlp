package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.{HasWordEmbeddings, ParamsAndFeaturesReadable}
import org.apache.spark.sql.SparkSession

trait EmbeddingsReadable[T <: HasWordEmbeddings] extends ParamsAndFeaturesReadable[T] {
  override def onRead(instance: T, path: String, spark: SparkSession): Unit = {
    instance.deserializeEmbeddings(path, spark.sparkContext)
  }
}
