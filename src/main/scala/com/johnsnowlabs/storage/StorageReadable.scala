package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.{HasFeatures, ParamsAndFeaturesReadable}
import org.apache.spark.sql.SparkSession

trait StorageReadable[T <: HasStorageModel with HasFeatures] extends ParamsAndFeaturesReadable[T] {

  def readStorage(instance: T, path: String, spark: SparkSession): Unit = {
    instance.deserializeStorage(path, spark)
  }

  addReader(readStorage)
}
