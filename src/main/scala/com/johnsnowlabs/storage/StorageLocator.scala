package com.johnsnowlabs.storage

import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

case class StorageLocator(database: String, storageRef: String, sparkSession: SparkSession) {

  private lazy val fileSystem = FileSystem.get(sparkSession.sparkContext.hadoopConfiguration)

  private val clusterTmpLocation: String = {
    val tmpLocation = getTmpLocation
    if (tmpLocation.startsWith("s3:/")) {
      tmpLocation
    } else {
      val tmpLocationPath = new Path(tmpLocation)
      fileSystem.mkdirs(tmpLocationPath)
      fileSystem.deleteOnExit(tmpLocationPath)
      tmpLocation
    }
  }

  val clusterFileName: String = {
    StorageHelper.resolveStorageName(database, storageRef)
  }

  val clusterFilePath: Path = {
    if (!getTmpLocation.startsWith("s3:/")) {
      Path.mergePaths(new Path(fileSystem.getUri.toString + clusterTmpLocation), new Path("/" + clusterFileName))
    } else new Path(clusterTmpLocation + "/" + clusterFileName)
  }

  val destinationScheme: String = if (getTmpLocation.startsWith("s3:/")) "s3" else fileSystem.getScheme

  private def getTmpLocation: String = {
    ConfigLoader.getConfigStringValue(ConfigHelper.storageTmpDir)
  }

}

object StorageLocator {
  def getStorageSerializedPath(path: String, folder: String, withinStorage: Boolean): Path =
    Path.mergePaths(new Path(path), new Path((if (withinStorage) "/storage/" else "/")+folder))
}