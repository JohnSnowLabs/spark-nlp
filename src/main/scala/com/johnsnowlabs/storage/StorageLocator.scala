package com.johnsnowlabs.storage

import com.johnsnowlabs.util.{ConfigHelperV2, ConfigLoaderV2}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

case class StorageLocator(database: String, storageRef: String, sparkSession: SparkSession) {

  private val fileSystem = FileSystem.get(sparkSession.sparkContext.hadoopConfiguration)

  private val clusterTmpLocation: String = {
//    val tmpLocation = ConfigHelper.getConfigValue(ConfigHelper.storageTmpDir).map(p => new Path(p)).getOrElse(
//      sparkSession.sparkContext.hadoopConfiguration.get("hadoop.tmp.dir")
//    ).toString+"/"+UUID.randomUUID().toString.takeRight(12)+"_cdx"
    val tmpLocation = ConfigLoaderV2.getConfigStringValue(ConfigHelperV2.storageTmpDir)
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