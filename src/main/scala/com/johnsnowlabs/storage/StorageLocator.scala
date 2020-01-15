package com.johnsnowlabs.storage

import com.johnsnowlabs.util.ConfigHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

case class StorageLocator(database: String, storageRef: String, sparkSession: SparkSession, fs: FileSystem) {

  private val clusterTmpLocation: Comparable[_] = {
    ConfigHelper.getConfigValue(ConfigHelper.storageTmpDir).map(p => new Path(p)).getOrElse(
      sparkSession.sparkContext.hadoopConfiguration.get("hadoop.tmp.dir")
    )
  }

  val clusterFileName: String = {
    StorageHelper.resolveStorageName(database, storageRef)
  }

  val clusterFilePath: Path = {
    Path.mergePaths(new Path(fs.getUri.toString + clusterTmpLocation), new Path("/"+clusterFileName))
  }

  val destinationScheme: String = {
    new Path(clusterFileName).getFileSystem(sparkSession.sparkContext.hadoopConfiguration).getScheme
  }

}

object StorageLocator {
  def getStorageSerializedPath(path: String, folder: String): Path =
    Path.mergePaths(new Path(path), new Path("/storage/"+folder))
}