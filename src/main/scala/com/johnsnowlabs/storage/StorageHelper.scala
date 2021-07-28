package com.johnsnowlabs.storage

import java.io.File

import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.{SparkContext, SparkFiles}
import org.apache.spark.sql.SparkSession


object StorageHelper {

  def resolveStorageName(database: String, storageRef: String): String = new Path(database + "_" + storageRef).toString

  def load(
            storageSourcePath: String,
            spark: SparkSession,
            database: String,
            storageRef: String,
            withinStorage: Boolean
          ): RocksDBConnection = {

    val dbFolder = StorageHelper.resolveStorageName(database, storageRef)
    val src = StorageLocator.getStorageSerializedPath(storageSourcePath.replaceAllLiterally("\\", "/"), dbFolder, withinStorage)

    val locator = StorageLocator(database, storageRef, spark)

    sendToCluster(src, locator.clusterFilePath, locator.clusterFileName, locator.destinationScheme, spark.sparkContext)

    RocksDBConnection.getOrCreate(locator.clusterFileName)
  }

  def save(path: String, connection: RocksDBConnection, spark: SparkSession, withinStorage: Boolean): Unit = {
    val indexUri = "file://"+(new java.net.URI(connection.findLocalIndex.replaceAllLiterally("\\", "/")).getPath)
    val index = new Path(indexUri)

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = new Path(path+{if (withinStorage) "/storage/" else ""})

    save(fs, index, dst)
  }

  private def save(fs: FileSystem, index: Path, dst: Path): Unit = {
    if (!fs.exists(dst))
      fs.mkdirs(dst)
    fs.copyFromLocalFile(false, true, index, dst)
  }

  def sendToCluster(source: Path, clusterFilePath: Path, clusterFileName: String, destinationScheme: String,
                    sparkContext: SparkContext): Unit = {
    destinationScheme match {
      case "file" => copyIndexToLocal(source, new Path(RocksDBConnection.getLocalPath(clusterFileName)), sparkContext)
      case _ => copyIndexToCluster(source, clusterFilePath, sparkContext)
    }
  }

  private def copyIndexToCluster(sourcePath: Path, dst: Path, spark: SparkContext): String = {
    if (!new File(SparkFiles.get(dst.getName)).exists()) {
      val srcFS = sourcePath.getFileSystem(spark.hadoopConfiguration)
      val dstFS = dst.getFileSystem(spark.hadoopConfiguration)

      if (srcFS.getScheme == "file") {
        val src = sourcePath
        dstFS.copyFromLocalFile(false, true, src, dst)
      } else {
        FileUtil.copy(srcFS, sourcePath, dstFS, dst, false, true, spark.hadoopConfiguration)
      }

      spark.addFile(dst.toString, recursive = true)
    }
    dst.toString
  }

  private def copyIndexToLocal(source: Path, destination: Path, context: SparkContext): Unit = {
    /** if we don't do a copy, and just move, it will all fail when re-saving utilized storage because of bad crc */
    val fs = source.getFileSystem(context.hadoopConfiguration)
    if (!fs.exists(destination))
      fs.copyFromLocalFile(false, true, source, destination)
  }

}