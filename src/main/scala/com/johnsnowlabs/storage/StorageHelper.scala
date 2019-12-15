package com.johnsnowlabs.storage

import java.nio.file.Files

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession


object StorageHelper {

  def load(
            storageSourcePath: String,
            spark: SparkSession,
            database: String
          ): RocksDBConnection = {

    val uri = new java.net.URI(storageSourcePath.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

//    val tmpFile = Files.createTempFile(database+"_", ".rdb").toAbsolutePath.toString
    val tmpDir = Files.createTempDirectory(database+"_").toAbsolutePath.toString

    fs.copyToLocalFile(new Path(storageSourcePath), new Path(tmpDir))
    val fileName = new Path(storageSourcePath).getName

    RocksDBConnection.getOrCreate(fileName)
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

