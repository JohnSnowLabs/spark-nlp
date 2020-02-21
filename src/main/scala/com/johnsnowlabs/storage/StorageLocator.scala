package com.johnsnowlabs.storage

import java.util.UUID

import com.johnsnowlabs.util.ConfigHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

case class StorageLocator(database: String, storageRef: String, sparkSession: SparkSession) {

  private val fs = FileSystem.get(sparkSession.sparkContext.hadoopConfiguration)

  private val clusterTmpLocation: String = {
    val tmpLocation = ConfigHelper.getConfigValue(ConfigHelper.storageTmpDir).map(p => new Path(p)).getOrElse(
      sparkSession.sparkContext.hadoopConfiguration.get("hadoop.tmp.dir")
    ).toString+"/"+UUID.randomUUID().toString.takeRight(12)+"_cdx"
    val tmpLocationPath = new Path(tmpLocation)
    fs.mkdirs(tmpLocationPath)
    fs.deleteOnExit(tmpLocationPath)
    tmpLocation
  }

  val clusterFileName: String = {
    StorageHelper.resolveStorageName(database, storageRef)
  }

  val clusterFilePath: Path = {
    Path.mergePaths(new Path(fs.getUri.toString + clusterTmpLocation), new Path("/"+clusterFileName))
  }

  val destinationScheme: String = {
    fs.getScheme
  }

}

object StorageLocator {
  def getStorageSerializedPath(path: String, folder: String, withinStorage: Boolean): Path =
    Path.mergePaths(new Path(path), new Path((if (withinStorage) "/storage/" else "/")+folder))
}