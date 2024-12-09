/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.storage

import com.johnsnowlabs.client.CloudResources
import com.johnsnowlabs.client.util.CloudHelper
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkContext, SparkFiles}

import java.io.File

object StorageHelper {

  def resolveStorageName(database: String, storageRef: String): String =
    new Path(database + "_" + storageRef).toString

  def load(
      storageSourcePath: String,
      spark: SparkSession,
      database: String,
      storageRef: String,
      withinStorage: Boolean): RocksDBConnection = {
    val dbFolder = StorageHelper.resolveStorageName(database, storageRef)
    val source = StorageLocator.getStorageSerializedPath(
      storageSourcePath.replaceAllLiterally("\\", "/"),
      dbFolder,
      withinStorage)

    val locator = StorageLocator(database, storageRef, spark)
    sendToCluster(
      source,
      locator.clusterFilePath,
      locator.clusterFileName,
      locator.destinationScheme,
      spark.sparkContext)

    val storagePath = if (locator.clusterFilePath.toString.startsWith("file:")) {
      locator.clusterFilePath.toString
    } else locator.clusterFileName

    RocksDBConnection.getOrCreate(storagePath)
  }

  def save(
      path: String,
      connection: RocksDBConnection,
      spark: SparkSession,
      withinStorage: Boolean): Unit = {
    val indexUri = "file://" + (new java.net.URI(
      connection.findLocalIndex.replaceAllLiterally("\\", "/")).getPath)
    val index = new Path(indexUri)

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = new Path(path + {
      if (withinStorage) "/storage/" else ""
    })

    save(fs, index, dst)
  }

  private def save(fs: FileSystem, index: Path, dst: Path): Unit = {
    if (!fs.exists(dst))
      fs.mkdirs(dst)
    fs.copyFromLocalFile(false, true, index, dst)
  }

  def sendToCluster(
      source: Path,
      clusterFilePath: Path,
      clusterFileName: String,
      destinationScheme: String,
      sparkContext: SparkContext): Unit = {
    destinationScheme match {
      case "file" => {
        val sourceFileSystemScheme = source.getFileSystem(sparkContext.hadoopConfiguration)
        val tmpIndexStorageLocalPath =
          RocksDBConnection.getTmpIndexStorageLocalPath(clusterFileName)
        sourceFileSystemScheme.getScheme match {
          case "file" => {
            if (!doesDirectoryExistJava(tmpIndexStorageLocalPath) ||
              !doesDirectoryExistHadoop(tmpIndexStorageLocalPath, sparkContext)) {
              copyIndexToLocal(source, new Path(tmpIndexStorageLocalPath), sparkContext)
            }
          }
          case "s3a" =>
            copyIndexToLocal(source, new Path(tmpIndexStorageLocalPath), sparkContext)
          case _ => {
            copyIndexToCluster(source, clusterFilePath, sparkContext)
          }
        }
      }
      case "abfss" =>
        if (clusterFilePath.toString.startsWith("file:")) {
          val tmpIndexStorageLocalPath =
            RocksDBConnection.getTmpIndexStorageLocalPath(clusterFileName)
          copyIndexToCluster(source, new Path("file://" + tmpIndexStorageLocalPath), sparkContext)
        } else {
          copyIndexToLocal(source, clusterFilePath, sparkContext)
        }
      case _ => {
        copyIndexToCluster(source, clusterFilePath, sparkContext)
      }
    }
  }

  private def doesDirectoryExistJava(path: String): Boolean = {
    val directory = new File(path)
    directory.exists && directory.isDirectory
  }

  private def doesDirectoryExistHadoop(path: String, sparkContext: SparkContext): Boolean = {
    val localPath = new Path(path)
    val fileSystem = localPath.getFileSystem(sparkContext.hadoopConfiguration)
    fileSystem.exists(localPath)
  }

  private def copyIndexToCluster(
      sourcePath: Path,
      dst: Path,
      sparkContext: SparkContext): String = {
    val destinationInSpark = new File(SparkFiles.get(dst.getName)).exists()
    if (!destinationInSpark) {
      val srcFS = sourcePath.getFileSystem(sparkContext.hadoopConfiguration)
      val dstFS = dst.getFileSystem(sparkContext.hadoopConfiguration)

      if (srcFS.getScheme == "file") {
        val src = sourcePath
        dstFS.copyFromLocalFile(false, true, src, dst)
      } else {
        FileUtil.copy(
          srcFS,
          sourcePath,
          dstFS,
          dst,
          false,
          true,
          sparkContext.hadoopConfiguration)
      }

      if (!CloudHelper.isMicrosoftFabric) {
        sparkContext.addFile(dst.toString, recursive = true)
      }
    }
    dst.toString
  }

  private def copyIndexToLocal(
      source: Path,
      destination: Path,
      sparkContext: SparkContext): Unit = {

    /** if we don't do a copy, and just move, it will all fail when re-saving utilized storage
      * because of bad crc
      */
    val fileSystemDestination = destination.getFileSystem(sparkContext.hadoopConfiguration)
    val fileSystemSource = source.getFileSystem(sparkContext.hadoopConfiguration)

    if (fileSystemSource.getScheme == "file") {
      fileSystemDestination.copyFromLocalFile(false, true, source, destination)
    } else {
      CloudResources.downloadBucketToLocalTmp(
        source.toString,
        destination.toString,
        isIndex = true)
      val isLocalMode = sparkContext.master.startsWith("local")
      if (isLocalMode) {
        sparkContext.addFile(destination.toString, recursive = true)
      }
    }
  }

}
