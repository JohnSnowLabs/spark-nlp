package com.johnsnowlabs.storage

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

trait HasStorageModel extends HasStorageRef with HasStorageReader {

  def serializeStorage(path: String, spark: SparkSession): Unit = {
    databases.foreach(database => {
      val index = new Path(RocksDBConnection.getLocalPath(getReader(database).getConnection.getFileName))

      val uri = new java.net.URI(path)
      val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
      val dst = StorageLocator.getStorageSerializedPath(path)

      StorageHelper.save(fs, index, dst)
    })
  }

  override protected def onWrite(path: String, spark: SparkSession): Unit = {
    serializeStorage(path, spark)
  }

  def deserializeStorage(path: String, spark: SparkSession): Unit = {
    val src = StorageLocator.getStorageSerializedPath(path)
    databases.foreach(database =>
    StorageHelper.load(
      src.toUri.toString,
      spark,
      database.toString
    ))
  }

}
