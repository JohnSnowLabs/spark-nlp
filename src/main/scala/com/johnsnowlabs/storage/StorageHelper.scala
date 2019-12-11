package com.johnsnowlabs.storage

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession


object StorageHelper {

  def load(
            indexPath: String
          ): RocksDBConnection = {
    RocksDBConnection.getOrCreate(indexPath)
  }

  def save(path: String, connection: RocksDBConnection, spark: SparkSession): Unit = {
    StorageHelper.save(path, spark, connection.getFileName)
  }

  private def save(path: String, spark: SparkSession, fileName: String): Unit = {
    val index = new Path(RocksDBConnection.getLocalPath(fileName))

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = new Path(path)

    save(fs, index, dst)
  }

  def save(fs: FileSystem, index: Path, dst: Path): Unit = {
    fs.copyFromLocalFile(false, true, index, dst)
  }



}

