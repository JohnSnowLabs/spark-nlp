package com.johnsnowlabs.storage

import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

case class StorageLocator(database: String, storageRef: String, sparkSession: SparkSession) {

  private val fileSystem = FileSystem.get(sparkSession.sparkContext.hadoopConfiguration)

  private val clusterTmpLocation: String = {
    val tmpLocation = ConfigLoader.getConfigStringValue(ConfigHelper.storageTmpDir)
    val tmpLocationPath = new Path(tmpLocation)
    fileSystem.mkdirs(tmpLocationPath)
    fileSystem.deleteOnExit(tmpLocationPath)
    tmpLocation
  }

  val clusterFileName: String = {
    StorageHelper.resolveStorageName(database, storageRef)
  }

  val clusterFilePath: Path = {
    Path.mergePaths(new Path(fileSystem.getUri.toString + clusterTmpLocation), new Path("/" + clusterFileName))
  }

  val destinationScheme: String = {
    fileSystem.getScheme
  }

}

object StorageLocator {
  def getStorageSerializedPath(path: String, folder: String, withinStorage: Boolean): Path =
    Path.mergePaths(new Path(path), new Path((if (withinStorage) "/storage/" else "/")+folder))
}