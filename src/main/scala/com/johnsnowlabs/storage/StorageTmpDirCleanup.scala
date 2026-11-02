/*
 * Copyright 2017-2026 John Snow Labs
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

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.scheduler.{SparkListener, SparkListenerApplicationEnd}
import org.slf4j.{Logger, LoggerFactory}

import java.nio.file.AccessDeniedException
import scala.collection.mutable
import scala.util.control.NonFatal

/** Deletes the S3 storage folders registered by [[StorageLocator]] when the application ends.
  * Other filesystems use Hadoop's deleteOnExit, which cannot take an S3 path.
  */
object StorageTmpDirCleanup {

  private val logger: Logger = LoggerFactory.getLogger(this.getClass.getName)

  private val registered = mutable.LinkedHashSet[Path]()
  private var listeningApplicationId: Option[String] = None

  /** Must not throw, it runs while a model is loading. */
  def register(path: Path, sparkContext: SparkContext): Unit = synchronized {
    registered += path
    if (!listeningApplicationId.contains(sparkContext.applicationId)) {
      try {
        sparkContext.addSparkListener(new CleanupListener(sparkContext.hadoopConfiguration))
        listeningApplicationId = Some(sparkContext.applicationId)
      } catch {
        case NonFatal(e) =>
          logger.warn(
            s"Could not schedule deletion of Spark NLP temporary storage folder $path: " +
              s"${e.getClass.getName}: ${e.getMessage}")
      }
    }
  }

  private[storage] def pending: Seq[Path] = synchronized { registered.toList }

  /** Executors can fetch the index until the application ends, so nothing is deleted earlier. */
  private[storage] class CleanupListener(hadoopConf: Configuration) extends SparkListener {
    override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {
      StorageTmpDirCleanup.synchronized { listeningApplicationId = None }
      cleanup(hadoopConf)
    }
  }

  /** Returns the folders that could not be deleted. */
  private[storage] def cleanup(hadoopConf: Configuration): Seq[Path] = {
    val paths = synchronized {
      val toDelete = registered.toList
      registered.clear()
      toDelete
    }
    paths.filterNot { path =>
      try {
        val fileSystem = path.getFileSystem(hadoopConf)
        if (fileSystem.delete(path, true)) {
          logger.info(s"Deleted Spark NLP temporary storage folder $path")
          true
        } else if (fileSystem.exists(path)) {
          logger.warn(s"Could not delete Spark NLP temporary storage folder $path")
          false
        } else true
      } catch {
        case NonFatal(e) if isPermissionDenied(e) =>
          logger.warn(
            s"Could not delete Spark NLP temporary storage folder $path: access denied. Check " +
              "s3:DeleteObject on the folder and s3:ListBucket on the bucket. " +
              s"Cause: ${e.getClass.getName}: ${e.getMessage}")
          false
        case NonFatal(e) =>
          logger.warn(
            s"Could not delete Spark NLP temporary storage folder $path: " +
              s"${e.getClass.getName}: ${e.getMessage}")
          logger.debug(s"Deleting $path failed", e)
          false
      }
    }
  }

  /** S3A throws AccessDeniedException; EMRFS wraps the AWS 403 message in an IOException. */
  private[storage] def isPermissionDenied(e: Throwable): Boolean =
    Iterator.iterate(e)(_.getCause).takeWhile(_ != null).take(10).exists {
      case _: AccessDeniedException => true
      case cause =>
        Option(cause.getMessage).exists(message =>
          message.contains("Access Denied") || message.contains("Status Code: 403"))
    }

}
