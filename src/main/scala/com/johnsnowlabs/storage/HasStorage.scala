/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

import java.nio.file.{Files, Paths, StandardCopyOption}
import java.util.UUID
import com.amazonaws.auth.DefaultAWSCredentialsProviderChain
import com.johnsnowlabs.nlp.HasCaseSensitiveProperties
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader, FileHelper}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Dataset, SparkSession}

trait HasStorage extends HasStorageRef with HasExcludableStorage with HasCaseSensitiveProperties {

  protected val databases: Array[Database.Name]

  val storagePath = new ExternalResourceParam(this, "storagePath", "path to file")

  def setStoragePath(path: String, readAs: String): this.type = set(storagePath, new ExternalResource(path, readAs, Map.empty[String, String]))

  def setStoragePath(path: String, readAs: ReadAs.Value): this.type = setStoragePath(path, readAs.toString)

  def getStoragePath: Option[ExternalResource] = get(storagePath)

  protected val missingRefMsg: String = s"Please set storageRef param in $this."

  protected def index(
                       fitDataset: Dataset[_],
                       storageSourcePath: Option[String],
                       readAs: Option[ReadAs.Value],
                       writers: Map[Database.Name, StorageWriter[_]],
                       readOptions: Option[Map[String, String]] = None
                     ): Unit

  protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_]

  private def indexDatabases(
                              databases: Array[Database.Name],
                              resource: Option[ExternalResource],
                              localFiles: Array[String],
                              fitDataset: Dataset[_],
                              spark: SparkContext
                            ): Unit = {

    require(databases.length == localFiles.length, "Storage temp locations must be equal to the amount of databases")

    lazy val connections = databases.zip(localFiles)
      .map { case (database, localFile) => (database, RocksDBConnection.getOrCreate(localFile)) }

    val writers = connections.map { case (db, conn) =>
      (db, createWriter(db, conn))
    }.toMap[Database.Name, StorageWriter[_]]

    val storageSourcePath = resource.map(r => importIfS3(r.path, spark).toUri.toString)
    if (resource.isDefined && new Path(resource.get.path).getFileSystem(spark.hadoopConfiguration).getScheme != "file") {
      val uri = new java.net.URI(storageSourcePath.get.replaceAllLiterally("\\", "/"))
      val fs = FileSystem.get(uri, spark.hadoopConfiguration)
      /** ToDo: What if the file is too large to copy to local? Index directly from hadoop? */
      val tmpFile = Files.createTempFile("sparknlp_", ".str").toAbsolutePath.toString
      fs.copyToLocalFile(new Path(storageSourcePath.get), new Path(tmpFile))
      index(fitDataset, Some(tmpFile), resource.map(_.readAs), writers, resource.map(_.options))
      FileHelper.delete(tmpFile)
    } else {
      index(fitDataset, storageSourcePath, resource.map(_.readAs), writers, resource.map(_.options))
    }

    writers.values.foreach(_.close())
    connections.map(_._2).foreach(_.close())
  }

  private def preload(
                       fitDataset: Dataset[_],
                       resource: Option[ExternalResource],
                       spark: SparkSession,
                       databases: Array[Database.Name]
                     ): Unit = {

    val sparkContext = spark.sparkContext

    val tmpLocalDestinations = {
      databases.map(_ =>
        Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_idx")
          .toAbsolutePath.toString
      )
    }

    indexDatabases(databases, resource, tmpLocalDestinations, fitDataset, sparkContext)

    val locators = databases.map(database => StorageLocator(database.toString, $(storageRef), spark))

    tmpLocalDestinations.zip(locators).foreach { case (tmpLocalDestination, locator) =>

      /** tmpFiles indexed must be explicitly set to be local files */
      val uri = "file://" + (new java.net.URI(tmpLocalDestination.replaceAllLiterally("\\", "/")).getPath)
      StorageHelper.sendToCluster(new Path(uri), locator.clusterFilePath, locator.clusterFileName, locator.destinationScheme, sparkContext)
    }

    // 3. Create Spark Embeddings
    locators.foreach(locator => RocksDBConnection.getOrCreate(locator.clusterFileName))
  }


  private def importIfS3(path: String, spark: SparkContext): Path = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    var src = new Path(path)
    //if the path contains s3a download to local cache if not present
    if (uri.getScheme != null) {
      if (uri.getScheme.equals("s3a")) {
        var accessKeyId = ConfigLoader.getConfigStringValue(ConfigHelper.accessKeyId)
        var secretAccessKey = ConfigLoader.getConfigStringValue(ConfigHelper.secretAccessKey)

        if (accessKeyId == "" || secretAccessKey == "") {
          val defaultCredentials = new DefaultAWSCredentialsProviderChain().getCredentials
          accessKeyId = defaultCredentials.getAWSAccessKeyId
          secretAccessKey = defaultCredentials.getAWSSecretKey
        }
        var old_key = ""
        var old_secret = ""
        if (spark.hadoopConfiguration.get("fs.s3a.access.key") != null) {
          old_key = spark.hadoopConfiguration.get("fs.s3a.access.key")
          old_secret = spark.hadoopConfiguration.get("fs.s3a.secret.key")
        }
        try {
          val dst = new Path(ResourceDownloader.cacheFolder, src.getName)
          if (!Files.exists(Paths.get(dst.toUri.getPath))) {
            //download s3 resource locally using config keys
            spark.hadoopConfiguration.set("fs.s3a.access.key", accessKeyId)
            spark.hadoopConfiguration.set("fs.s3a.secret.key", secretAccessKey)
            val s3fs = FileSystem.get(uri, spark.hadoopConfiguration)

            val dst_tmp = new Path(ResourceDownloader.cacheFolder, src.getName + "_tmp")


            s3fs.copyToLocalFile(src, dst_tmp)
            // rename to original file
            Files.move(
              Paths.get(dst_tmp.toUri.getRawPath),
              Paths.get(dst.toUri.getRawPath),
              StandardCopyOption.REPLACE_EXISTING
            )

          }
          src = new Path(dst.toUri.getPath)
        }
        finally {
          //reset the keys
          if (!old_key.equals("")) {
            spark.hadoopConfiguration.set("fs.s3a.access.key", old_key)
            spark.hadoopConfiguration.set("fs.s3a.secret.key", old_secret)
          }
        }

      }
    }
    src
  }

  private var preloaded = false

  def indexStorage(fitDataset: Dataset[_], resource: Option[ExternalResource]): Unit = {
    if (!preloaded) {
      preloaded = true
      require(isDefined(storageRef), missingRefMsg)
      preload(
        fitDataset,
        resource,
        fitDataset.sparkSession,
        databases
      )
    }
  }

}
