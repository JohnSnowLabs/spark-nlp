package com.johnsnowlabs.storage

import org.apache.spark.sql.SparkSession

trait HasStorageModel extends HasStorageReader {

  protected val databases: Array[Database.Name]

  def serializeStorage(path: String, spark: SparkSession): Unit = {
    databases.foreach(database => {
      StorageHelper.save(path, getReader(database).getConnection, spark)
    })
  }

  override protected def onWrite(path: String, spark: SparkSession): Unit = {
    serializeStorage(path, spark)
  }

  def deserializeStorage(path: String, spark: SparkSession): Unit = {
    databases.foreach(database => {
      val dbFolder = StorageHelper.resolveStorageName(database.toString, $(storageRef))
      val src = StorageLocator.getStorageSerializedPath(path, dbFolder)
      StorageHelper.load(
        src.toUri.toString,
        spark,
        database.toString,
        $(storageRef)
      )
    })
  }

}
