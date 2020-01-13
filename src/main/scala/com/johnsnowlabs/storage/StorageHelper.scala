package com.johnsnowlabs.storage

import com.johnsnowlabs.util.FileHelper
import org.apache.hadoop.fs.{FileSystem, Path}
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

    sendToCluster(storageSourcePath, locator.clusterFilePath, locator.clusterFileName, locator.destinationScheme, spark.sparkContext, isIndexed = true)

    RocksDBConnection.getOrCreate(locator.clusterFileName)
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

  def sendToCluster(localDestination: String, clusterFilePath: Path, clusterFileName: String, destinationScheme: String, sparkContext: SparkContext, isIndexed: Boolean): Unit = {
    if (destinationScheme == "file") {
      copyIndexToLocal(new Path(localDestination), new Path(RocksDBConnection.getLocalPath(clusterFileName)), sparkContext, isIndexed)
    } else {
      copyIndexToCluster(localDestination, clusterFilePath, sparkContext)
      FileHelper.delete(localDestination)
    }
  }

  private def copyIndexToCluster(localFile: String, dst: Path, spark: SparkContext): String = {
    val fs = new Path(localFile).getFileSystem(spark.hadoopConfiguration)
    val src = new Path(localFile)

    /** This fails if working on local file system, because spark.addFile will detect simoultaneous writes on same location and fail */
    fs.copyFromLocalFile(false, true, src, dst)
    fs.deleteOnExit(dst)

    spark.addFile(dst.toString, true)

    dst.toString
  }

  private def copyIndexToLocal(source: Path, destination: Path, context: SparkContext, isIndexed: Boolean) = {
    /** if we don't do a copy, and just move, it will all fail when re-saving utilized storage because of bad crc */
    val fs = source.getFileSystem(context.hadoopConfiguration)
    fs.copyFromLocalFile(false, true, source, if (isIndexed) destination.getParent else destination)
  }

}