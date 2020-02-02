package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.{HasFeatures, ParamsAndFeaturesReadable}
import org.apache.spark.sql.SparkSession

trait StorageReadable[T <: HasStorageModel with HasFeatures] extends ParamsAndFeaturesReadable[T] {

  val databases: Array[Database.Name]

  def loadStorage(path: String, spark: SparkSession, storageRef: String): Unit = {
    databases.foreach(database => {
      StorageHelper.load(
        path,
        spark,
        database.toString,
        storageRef,
        withinStorage = false
      )
    })
  }

  def readStorage(instance: T, path: String, spark: SparkSession): Unit = {
    instance.deserializeStorage(path, spark)
  }

  addReader(readStorage)
}
