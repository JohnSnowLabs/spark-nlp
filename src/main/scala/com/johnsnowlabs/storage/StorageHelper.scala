package com.johnsnowlabs.storage

import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession


object StorageHelper {

  def resolveStorageName(database: String, storageRef: String): String = new Path(database + "_" + storageRef).toString

  def load(
            storageSourcePath: String,
            spark: SparkSession,
            database: String,
            storageRef: String
          ): RocksDBConnection = {

    val uri = new java.net.URI(storageSourcePath.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    val locator = StorageLocator(database, storageRef, spark, fs)

    sendToCluster(new Path(storageSourcePath), locator.clusterFilePath, locator.clusterFileName, locator.destinationScheme, spark.sparkContext)

    RocksDBConnection.getOrCreate(locator.clusterFileName)
  }

  def save(path: String, connection: RocksDBConnection, spark: SparkSession): Unit = {
    StorageHelper.save(path, spark, connection)
  }

  private def save(path: String, spark: SparkSession, connection: RocksDBConnection): Unit = {
    val index = new Path("file://"+connection.findLocalIndex).getParent

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = new Path(path+"/storage/")

    save(fs, index, dst)
  }

  private def save(fs: FileSystem, index: Path, dst: Path): Unit = {
    fs.copyFromLocalFile(false, true, index, dst)
  }

  def sendToCluster(source: Path, clusterFilePath: Path, clusterFileName: String, destinationScheme: String, sparkContext: SparkContext): Unit = {
    if (destinationScheme == "file") {
      copyIndexToLocal(source, new Path(RocksDBConnection.getLocalPath(clusterFileName)), sparkContext)
    } else {
      copyIndexToCluster(source, clusterFilePath, sparkContext)
    }
  }

  private def copyIndexToCluster(sourcePath: Path, dst: Path, spark: SparkContext): String = {
    val fs = sourcePath.getFileSystem(spark.hadoopConfiguration)
    if (fs.getScheme == "file") {
      val src = sourcePath
      dst.getFileSystem(spark.hadoopConfiguration).copyFromLocalFile(false, true, src, dst)
    } else if (!fs.exists(dst)) {
      FileUtil.copy(fs, sourcePath, fs, dst, false, true, spark.hadoopConfiguration)
    }

    spark.addFile(dst.toString, recursive = true)

    dst.toString
  }

  private def copyIndexToLocal(source: Path, destination: Path, context: SparkContext): Unit = {
    /** if we don't do a copy, and just move, it will all fail when re-saving utilized storage because of bad crc */
    val fs = source.getFileSystem(context.hadoopConfiguration)
    if (!fs.exists(destination))
      fs.copyFromLocalFile(false, true, source, destination)
  }

}