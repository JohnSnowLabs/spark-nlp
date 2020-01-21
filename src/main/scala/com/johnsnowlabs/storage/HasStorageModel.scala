package com.johnsnowlabs.storage

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

trait HasStorageModel extends HasStorageReader {

  protected val databases: Array[Database.Name]

  def serializeStorage(path: String, spark: SparkSession): Unit = {
    databases.foreach(database => {
      val databaseFileName = getReader(database).getConnection.getFileName
      val source = RocksDBConnection.getLocalPath(databaseFileName)
      val index = new Path(source)

      val uri = new java.net.URI(path)
      val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
      val dst = StorageLocator.getStorageSerializedPath(path, databaseFileName)

      StorageHelper.save(fs, index, dst)
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
