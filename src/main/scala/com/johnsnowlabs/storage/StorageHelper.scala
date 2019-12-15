package com.johnsnowlabs.storage

import java.nio.file.Files

import com.johnsnowlabs.util.FileHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession


object StorageHelper {

  def load(
            storageSourcePath: String,
            spark: SparkSession,
            database: String
          ): RocksDBConnection = {

    val uri = new java.net.URI(storageSourcePath.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    val tmpDir = Files.createTempDirectory(database+"_").toAbsolutePath.toString

    fs.copyToLocalFile(new Path(storageSourcePath), new Path(tmpDir))

    val locator = StorageLocator(database, spark, fs)

    sendToCluster(tmpDir, locator.clusterFilePath, locator.clusterFileName, locator.destinationScheme, spark.sparkContext)

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

  def sendToCluster(tmpLocalDestination: String, clusterFilePath: Path, clusterFileName: String, destinationScheme: String, sparkContext: SparkContext): Unit = {
    if (destinationScheme == "file") {
      copyIndexToLocal(new Path(tmpLocalDestination), new Path(RocksDBConnection.getLocalPath(clusterFileName)), sparkContext)
    } else {
      copyIndexToCluster(tmpLocalDestination, clusterFilePath, sparkContext)
      FileHelper.delete(tmpLocalDestination)
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

  private def copyIndexToLocal(source: Path, destination: Path, context: SparkContext) = {
    /** if we don't do a copy, and just move, it will all fail when re-saving utilized storage because of bad crc */
    val fs = source.getFileSystem(context.hadoopConfiguration)
    fs.copyFromLocalFile(false, true, source, destination)
    fs.deleteOnExit(source)
  }

}

