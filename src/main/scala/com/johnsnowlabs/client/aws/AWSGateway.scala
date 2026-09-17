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

package com.johnsnowlabs.client.aws

import com.amazonaws.auth.{AWSCredentials, AWSStaticCredentialsProvider}
import com.amazonaws.event.{ProgressEvent, ProgressEventType, ProgressListener}
import com.amazonaws.services.s3.model.{
  GetObjectRequest,
  ObjectMetadata,
  PutObjectResult,
  S3Object,
  S3ObjectSummary
}
import com.amazonaws.services.s3.transfer.{Transfer, TransferManagerBuilder}
import com.amazonaws.services.s3.{AmazonS3, AmazonS3ClientBuilder}
import com.amazonaws.{AmazonClientException, AmazonServiceException, ClientConfiguration}
import com.johnsnowlabs.client.CloudStorage
import com.johnsnowlabs.nlp.pretrained.ResourceMetadata
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.slf4j.{Logger, LoggerFactory}

import scala.jdk.CollectionConverters._
import java.io.{File, FileInputStream, InputStream, PrintStream, RandomAccessFile}
import java.nio.file.Files
import java.util.concurrent.{Executors, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong}
import scala.util.control.NonFatal

class AWSGateway(
    accessKeyId: String = ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalAccessKeyId),
    secretAccessKey: String =
      ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalSecretAccessKey),
    sessionToken: String =
      ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalSessionToken),
    awsProfile: String = ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalProfileName),
    region: String = ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalRegion),
    credentialsType: String = "private")
    extends AutoCloseable
    with CloudStorage {

  protected val logger: Logger = LoggerFactory.getLogger(this.getClass.toString)

  lazy val client: AmazonS3 = {
    if (region.isEmpty || region == null) {
      throw new Exception("Region argument is mandatory to create Amazon S3 client.")
    }
    var credentialParams =
      CredentialParams(accessKeyId, secretAccessKey, sessionToken, awsProfile, region)
    if (credentialsType == "public" || credentialsType == "community") {
      credentialParams = CredentialParams("anonymous", "", "", "", region)
    }
    val awsCredentials = new AWSTokenCredentials
    val credentials: Option[AWSCredentials] = awsCredentials.buildCredentials(credentialParams)

    getAmazonS3Client(credentials)
  }

  /** How many concurrent HTTP range requests a single object download may use.
    *
    * A `lazy val` rather than a `def`: ConfigLoader memoises its values, so this cannot change
    * during the life of the JVM, and computing it once means an out-of-range setting is reported
    * once rather than on every download.
    */
  private lazy val downloadThreads: Int = {
    val requested = ConfigLoader.getConfigIntValue(ConfigHelper.pretrainedDownloadThreads)
    val effective = math.min(AWSGateway.MaxParallelConnections, math.max(1, requested))
    if (requested > AWSGateway.MaxParallelConnections) {
      logger.warn(
        s"${ConfigHelper.pretrainedDownloadThreads} is set to $requested, which is above the " +
          s"supported maximum of ${AWSGateway.MaxParallelConnections}; using $effective " +
          "instead. The limit exists because this setting sizes two pools, not one: a thread " +
          "pool per download, and the HTTP connection pool of the S3 client shared by the whole " +
          "process -- so an unbounded value would hold that many sockets open for every S3 " +
          "call, not just for downloads. Throughput stops rewarding extra connections well " +
          "before this point in any case: past a few dozen, each range is too small to outlive " +
          "its own TLS handshake, so the added requests buy latency and S3 charges rather than " +
          "bandwidth.")
    } else if (requested < 1) {
      logger.warn(
        s"${ConfigHelper.pretrainedDownloadThreads} is set to $requested; using $effective " +
          "instead. A download needs at least one connection.")
    }
    effective
  }

  /** How many connections to actually open for an object of this size.
    *
    * `download_threads` is an upper bound, not a target: every worker is guaranteed at least
    * `MinRangeBytes` to move. Without that floor, a high setting on a small object opens dozens
    * of connections carrying a few tens of KB each, where opening the connection costs far more
    * than the bytes it delivers -- slower than a single connection, and billed as one S3 GET per
    * range.
    *
    * This also subsumes a separate "too small to split at all" threshold: anything under two
    * whole ranges resolves to one connection and takes the sequential path.
    */
  private def effectiveConnections(contentLength: Long): Int = {
    if (contentLength <= 0) 1
    else
      math.max(
        1,
        math.min(downloadThreads.toLong, contentLength / AWSGateway.MinRangeBytes).toInt)
  }

  private def getAmazonS3Client(credentials: Option[AWSCredentials]): AmazonS3 = {
    val config = new ClientConfiguration()
    val timeout = ConfigLoader.getConfigIntValue(ConfigHelper.s3SocketTimeout)
    config.setSocketTimeout(timeout)

    if (downloadThreads > config.getMaxConnections) {
      config.setMaxConnections(downloadThreads)
    }

    val s3Client = {
      if (credentials.isDefined) {
        AmazonS3ClientBuilder
          .standard()
          .withCredentials(new AWSStaticCredentialsProvider(credentials.get))
          .withClientConfiguration(config)
      } else {
        val warning_message =
          "Unable to build AWS credential via AWSGateway chain, some parameter is missing or" +
            " malformed. S3 integration may not work well."
        logger.warn(warning_message)
        AmazonS3ClientBuilder
          .standard()
          .withClientConfiguration(config)
      }
    }

    s3Client.withRegion(region).build()
  }

  override def doesBucketPathExist(bucketName: String, filePath: String): Boolean = {
    try {
      val listObjects = client.listObjectsV2(bucketName, filePath)
      listObjects.getObjectSummaries.size() > 0
    } catch {
      case exception: AmazonServiceException =>
        if (exception.getStatusCode == 404) false else throw exception
      case NonFatal(unexpectedException) =>
        val methodName = Thread.currentThread.getStackTrace()(1).getMethodName
        throw new Exception(
          s"Unexpected error in ${this.getClass.getName}.$methodName: $unexpectedException")
    }
  }

  override def copyFileToBucket(
      bucketName: String,
      destinationPath: String,
      inputStream: InputStream): Unit = {
    val metadata = new ObjectMetadata()
    inputStream match {
      case fileInputStream: FileInputStream =>
        metadata.setContentLength(fileInputStream.getChannel.size())
      case _ =>
    }
    client.putObject(bucketName, destinationPath, inputStream, metadata)
  }

  override def copyInputStreamToBucket(
      bucketName: String,
      filePath: String,
      sourceFilePath: String): Unit = {
    val fileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val inputStream = fileSystem.open(new Path(sourceFilePath))
    client.putObject(bucketName, filePath, inputStream, new ObjectMetadata())
  }

  def getMetadata(s3Path: String, folder: String, bucket: String): List[ResourceMetadata] = {
    val metaFile = getS3File(s3Path, folder, "metadata.json")
    val obj = this.client.getObject(bucket, metaFile)
    val metadata = ResourceMetadata.readResources(obj.getObjectContent)
    metadata
  }

  def getS3File(parts: String*): String = {
    parts
      .map(part => part.stripSuffix("/"))
      .filter(part => part.nonEmpty)
      .mkString("/")
  }

  /** A HEAD request. Returns `None` for a missing object, so callers can use it both as an
    * existence check and as the source of the content length a ranged download needs -- one round
    * trip instead of two.
    */
  def getS3ObjectMetadata(bucket: String, s3FilePath: String): Option[ObjectMetadata] = {
    try {
      Some(client.getObjectMetadata(bucket, s3FilePath))
    } catch {
      case exception: AmazonServiceException =>
        if (exception.getStatusCode == 404) None else throw exception
      case NonFatal(unexpectedException) =>
        val methodName = Thread.currentThread.getStackTrace()(1).getMethodName
        throw new Exception(
          s"Unexpected error in ${this.getClass.getName}.$methodName: $unexpectedException")
    }
  }

  /** Downloads an S3 object, splitting it across several parallel HTTP range requests.
    *
    * A single connection is throughput-limited on a high-latency link, where each TCP flow needs
    * a long time to open its congestion window; splitting the object into contiguous ranges
    * recovers most of that loss. Unlike a multipart-aware transfer, this works on any object
    * whatever way it was uploaded, and honours the configured connection count exactly rather
    * than being capped by the object's part count.
    *
    * Small objects keep to one connection, since the extra round trips cost more than they save.
    * Any failure falls back to the single-connection path rather than propagating, so this can
    * only be faster or equal, never a new source of download errors. Callers still validate the
    * checksum afterwards, so a truncated parallel download cannot pass silently.
    *
    * @param objectMetadata
    *   already-fetched metadata for this object, when the caller has it; supplying it avoids a
    *   second HEAD request for the content length.
    * @param showProgress
    *   whether to draw a progress bar. The bar is for downloads the user asked for by name; an
    *   internal fetch such as the models index should pass `false`, so the console does not
    *   report progress against a file nobody requested.
    */
  def getS3Object(
      bucket: String,
      s3FilePath: String,
      tmpFile: File,
      objectMetadata: Option[ObjectMetadata] = None,
      showProgress: Boolean = true): Unit = {
    val metadata = objectMetadata.orElse(getS3ObjectMetadata(bucket, s3FilePath))
    val contentLength = metadata.map(_.getContentLength).getOrElse(-1L)
    val label = s3FilePath.split("/").last
    val connections = effectiveConnections(contentLength)

    if (connections <= 1) {
      getS3ObjectSequentially(bucket, s3FilePath, tmpFile, contentLength, label, showProgress)
    } else if (!downloadRangesInParallel(
        bucket,
        s3FilePath,
        tmpFile,
        connections,
        contentLength,
        label,
        showProgress)) {
      logger.warn(
        s"Parallel download of $s3FilePath did not complete; " +
          s"falling back to a single connection.")
      tmpFile.delete()
      getS3ObjectSequentially(bucket, s3FilePath, tmpFile, contentLength, label, showProgress)
    }
  }

  /** The pre-existing single-connection path, kept as the fallback and for small objects. */
  private def getS3ObjectSequentially(
      bucket: String,
      s3FilePath: String,
      tmpFile: File,
      contentLength: Long,
      label: String,
      showProgress: Boolean): Unit = {
    val request = new GetObjectRequest(bucket, s3FilePath)
    if (showProgress && contentLength > 0) {
      request.setGeneralProgressListener(new DownloadProgress(contentLength, label).listener)
    }
    client.getObject(request, tmpFile)
  }

  /** @return true only when every range completed and the assembled file is the expected size */
  private def downloadRangesInParallel(
      bucket: String,
      s3FilePath: String,
      tmpFile: File,
      connections: Int,
      contentLength: Long,
      label: String,
      showProgress: Boolean): Boolean = {

    val ranges = AWSGateway.byteRanges(contentLength, connections)
    // byteRanges never returns more ranges than there are bytes, so a tiny object cannot spawn
    // a pool wider than the work available.
    val pool = Executors.newFixedThreadPool(ranges.size)
    val progress =
      if (showProgress) Some(new DownloadProgress(contentLength, label)) else None
    val written = new AtomicLong(0L)
    val failed = new AtomicBoolean(false)

    try {
      // Preallocate so each worker can seek straight to its own offset.
      val preallocate = new RandomAccessFile(tmpFile, "rw")
      try preallocate.setLength(contentLength)
      finally preallocate.close()

      ranges.foreach { case (start, end) =>
        pool.execute(new Runnable {
          override def run(): Unit = {
            var output: RandomAccessFile = null
            var input: InputStream = null
            try {
              output = new RandomAccessFile(tmpFile, "rw")
              output.seek(start)
              val request = new GetObjectRequest(bucket, s3FilePath).withRange(start, end)
              input = client.getObject(request).getObjectContent
              val buffer = new Array[Byte](AWSGateway.DownloadBufferBytes)
              var read = input.read(buffer)
              while (read > -1) {
                output.write(buffer, 0, read)
                written.addAndGet(read.toLong)
                progress.foreach(_.advance(read.toLong))
                read = input.read(buffer)
              }
            } catch {
              case NonFatal(e) =>
                failed.set(true)
                logger.warn(s"Range $start-$end of $s3FilePath failed: ${e.getMessage}")
            } finally {
              if (input != null) input.close()
              if (output != null) output.close()
            }
          }
        })
      }

      pool.shutdown()
      val finished =
        pool.awaitTermination(AWSGateway.ParallelDownloadTimeoutHours, TimeUnit.HOURS)
      val complete = finished && !failed.get() && written.get() == contentLength
      if (complete) progress.foreach(_.finish())
      complete
    } catch {
      case NonFatal(e) =>
        logger.warn(s"Parallel download of $s3FilePath failed: ${e.getMessage}")
        false
    } finally {
      pool.shutdownNow()
    }
  }

  def getS3Object(bucket: String, s3FilePath: String): S3Object = {
    val s3Object = client.getObject(bucket, s3FilePath)
    s3Object
  }

  def getS3DownloadSize(
      s3Path: String,
      folder: String,
      fileName: String,
      bucket: String): Option[Long] = {
    try {
      val s3FilePath = getS3File(s3Path, folder, fileName)
      val meta = client.getObjectMetadata(bucket, s3FilePath)
      Some(meta.getContentLength)
    } catch {
      case exception: AmazonServiceException =>
        if (exception.getStatusCode == 404) None else throw exception
      case NonFatal(unexpectedException) =>
        val methodName = Thread.currentThread.getStackTrace()(1).getMethodName
        throw new Exception(
          s"Unexpected error in ${this.getClass.getName}.$methodName: $unexpectedException")
    }
  }

  override def downloadFilesFromBucketToDirectory(
      bucketName: String,
      filePath: String,
      directoryPath: String,
      isIndex: Boolean = false): Unit = {

    val transferManager = TransferManagerBuilder
      .standard()
      .withS3Client(client)
      .build()
    try {
      val multipleFileDownload =
        transferManager.downloadDirectory(bucketName, filePath, new File(directoryPath))
      println(multipleFileDownload.getDescription)
      waitForCompletion(multipleFileDownload)
    } catch {
      case e: AmazonServiceException =>
        throw new AmazonServiceException(
          "Amazon service error when downloading files from S3 directory: " + e.getMessage)
    }
    transferManager.shutdownNow()

    if (isIndex) {
      // Recursively rename the downloaded files to the desired directory path
      def renameFiles(directory: File, keySuffix: String): Unit = {
        val downloadedFiles = directory.listFiles()
        for (file <- downloadedFiles) {
          if (file.isDirectory()) {
            // If the file is a directory, recursively rename its contents
            val subDirectory = new File(directory, file.getName())
            val subKeySuffix = file.getName()
            renameFiles(subDirectory, subKeySuffix)
          } else {
            // Otherwise, rename the file to the desired local file path
            val fileName = file.getName()
            val newFilePath =
              new File(directoryPath, fileName.stripPrefix(directoryPath)).getPath()
            file.renameTo(new File(newFilePath))
          }
        }
      }

      // Rename the downloaded files to the desired local file path
      val keySuffix = directoryPath.split("/").tail.mkString("/")
      renameFiles(new File(directoryPath), keySuffix)

      // Remove all old folders
      def removeAllFolders(directoryPath: File): Unit = {
        val files = directoryPath.listFiles()
        for (file <- files) {
          if (file.isDirectory()) {
            removeAllFolders(file)
            Files.deleteIfExists(file.toPath())
          }
        }
      }

      removeAllFolders(new File(directoryPath))
    }

  }

  private def waitForCompletion(transfer: Transfer): Unit = {
    try transfer.waitForCompletion()
    catch {
      case e: AmazonServiceException =>
        throw new AmazonServiceException("Amazon service error: " + e.getMessage)
      case e: AmazonClientException =>
        throw new AmazonClientException("Amazon client error: " + e.getMessage)
      case e: InterruptedException =>
        throw new InterruptedException("Transfer interrupted: " + e.getMessage)
    }
  }

  def listS3Files(bucket: String, s3Path: String): Array[S3ObjectSummary] = {
    try {
      val listObjects = client.listObjectsV2(bucket, s3Path)
      listObjects.getObjectSummaries.asScala.toArray
    } catch {
      case e: AmazonServiceException =>
        throw new AmazonServiceException("Amazon service error: " + e.getMessage)
      case NonFatal(unexpectedException) =>
        val methodName = Thread.currentThread.getStackTrace()(1).getMethodName
        throw new Exception(
          s"Unexpected error in ${this.getClass.getName}.$methodName: $unexpectedException")
    }
  }

  override def close(): Unit = {
    client.shutdown()
  }

}

/** How download progress is rendered, and how that is decided.
  *
  */
private[aws] object ProgressStyle {

  val Bar = "bar"
  val Lines = "lines"
  val Off = "off"

  /** Percent step between updates in [[Lines]] mode: one line per whole percent, so a download
    * produces at most 101 lines.
    */
  val LineStepPercent = 1L

  /** Resolved once per JVM: the setting cannot change while it runs, and probing the console on
    * every download would be wasted work.
    */
  lazy val resolved: String = fromConfig(
    ConfigLoader.getConfigStringValue(ConfigHelper.pretrainedDownloadProgress))

  private[aws] def fromConfig(setting: String): String =
    Option(setting).map(_.trim.toLowerCase).getOrElse("") match {
      case Bar => Bar
      case Lines => Lines
      case Off => Off
      // "auto", empty, or anything unrecognised: unknown input behaves like the default.
      case _ => if (isTerminal) Bar else Lines
    }

  /** Whether stdout is a real terminal.
    *
    */
  private[aws] def isTerminal: Boolean = {
    val console = System.console()
    if (console == null) false
    else
      try console.getClass.getMethod("isTerminal").invoke(console).asInstanceOf[Boolean]
      catch {
        case _: NoSuchMethodException => true // pre-22: a non-null Console is a real terminal
        case NonFatal(_) => false
      }
  }
}

/** Byte-accurate progress bar for one download.
  *
  */
private class DownloadProgress(
    totalBytes: Long,
    label: String,
    out: PrintStream = System.out,
    style: String = ProgressStyle.resolved) {

  private val transferred = new AtomicLong(0L)
  private val lastPercentDrawn = new AtomicLong(-1L)
  private var lastLineStep = -1L // guarded by drawLock
  private val drawLock = new Object

  def advance(bytes: Long): Unit = {
    if (bytes <= 0 || totalBytes <= 0) return
    val done = transferred.addAndGet(bytes)
    val percent = math.min(100L, (done * 100) / totalBytes)
    if (percent > lastPercentDrawn.get()) {
      drawLock.synchronized {
        if (percent > lastPercentDrawn.get()) {
          lastPercentDrawn.set(percent)
          draw(percent, done)
        }
      }
    }
  }

  def finish(): Unit = {
    if (totalBytes > 0 && lastPercentDrawn.get() < 100L) {
      lastPercentDrawn.set(100L)
      draw(100L, totalBytes)
    }
  }

  private def draw(percent: Long, done: Long): Unit = drawLock.synchronized {
    style match {
      case ProgressStyle.Off => ()
      case ProgressStyle.Bar =>
        val width = 30
        val filled = ((percent * width) / 100).toInt
        val bar = ("=" * filled) + (if (filled < width) ">"
                                    else "") + (" " * (width - filled - 1).max(0))
        out.print(
          f"\r  [$bar] $percent%3d%%  (${done / 1e6}%.1f / ${totalBytes / 1e6}%.1f MB) $label")
        out.flush()
        if (percent >= 100) out.println()
      case _ =>
        val step = percent / ProgressStyle.LineStepPercent
        if (step > lastLineStep) {
          lastLineStep = step
          out.println(
            f"  downloading $label ... $percent%3d%%  " +
              f"(${done / 1e6}%.1f / ${totalBytes / 1e6}%.1f MB)")
          out.flush()
        }
    }
  }

  /** Adapter so the single-connection path, which transfers inside the SDK, feeds the same bar.
    */
  def listener: ProgressListener = new ProgressListener {
    override def progressChanged(event: ProgressEvent): Unit = {
      if (event.getEventType == ProgressEventType.RESPONSE_BYTE_TRANSFER_EVENT) {
        advance(event.getBytesTransferred)
      }
    }
  }
}

object AWSGateway {

  /** Smallest slice worth giving a connection of its own.
    *
    */
  val MinRangeBytes: Long = 4L * 1024 * 1024

  val DownloadBufferBytes: Int = 256 * 1024

  val MaxParallelConnections: Int = 1000

  val ParallelDownloadTimeoutHours: Long = 12L

  private[aws] def byteRanges(contentLength: Long, connections: Int): Seq[(Long, Long)] = {
    require(contentLength > 0, "contentLength must be positive")
    require(connections > 0, "connections must be positive")
    val usable = math.min(connections.toLong, contentLength).toInt
    val chunkSize = contentLength / usable
    (0 until usable).map { index =>
      val start = index * chunkSize
      val end = if (index == usable - 1) contentLength - 1 else start + chunkSize - 1
      (start, end)
    }
  }
}
