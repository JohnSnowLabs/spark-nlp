package com.johnsnowlabs.storage

import org.apache.spark.sql.SparkSession

trait HasStorageModel extends HasStorageReader with HasExcludableStorage {

  protected val databases: Array[Database.Name]

  def serializeStorage(path: String, spark: SparkSession): Unit = {
    if ($(includeStorage))
      saveStorage(path, spark, withinStorage = true)
  }

  def saveStorage(path: String, spark: SparkSession, withinStorage: Boolean = false): Unit = {
    databases.foreach(database => {
      StorageHelper.save(path, getReader(database).getConnection, spark, withinStorage)
    })
  }

  override protected def onWrite(path: String, spark: SparkSession): Unit = {
    serializeStorage(path, spark)
  }

  def deserializeStorage(path: String, spark: SparkSession): Unit = {
    if ($(includeStorage))
      databases.foreach(database => {
        StorageHelper.load(
          path,
          spark,
          database.toString,
          $(storageRef),
          withinStorage = true
        )
      })
  }

}
