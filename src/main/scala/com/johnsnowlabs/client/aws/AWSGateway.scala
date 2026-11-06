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
  S3ObjectInputStream,
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
import java.io.{
  EOFException,
  File,
  FileInputStream,
  IOException,
  InputStream,
  PrintStream,
  RandomAccessFile
}
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Files, StandardCopyOption, StandardOpenOption}
import java.util.concurrent.{
  CancellationException,
  ConcurrentHashMap,
  Executors,
  ThreadFactory,
  TimeUnit
}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger, AtomicLong}
import scala.util.Try
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
  protected lazy val downloadThreads: Int = {
    val requested = ConfigLoader.getConfigIntValue(ConfigHelper.pretrainedDownloadThreads)
    val effective = AWSGateway.clampDownloadThreads(requested)
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

  private def effectiveConnections(contentLength: Long): Int =
    AWSGateway.connectionsFor(contentLength, downloadThreads)

  /** How long one ranged download may take before it is abandoned for a single connection. */
  protected def parallelDownloadTimeoutMillis: Long =
    TimeUnit.HOURS.toMillis(AWSGateway.ParallelDownloadTimeoutHours)

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

  /** Kept with its 7.0 signature for callers built against it. Downloads through the ranged path
    * and returns the object's metadata, as it always did, and like 7.0 shows no progress.
    */
  def getS3Object(bucket: String, s3FilePath: String, tmpFile: File): ObjectMetadata =
    getS3ObjectMetadata(bucket, s3FilePath) match {
      case Some(metadata) =>
        getS3Object(bucket, s3FilePath, tmpFile, Some(metadata), showProgress = false)
        metadata
      // A missing object: let the SDK raise exactly what this method raised before.
      case None => client.getObject(new GetObjectRequest(bucket, s3FilePath), tmpFile)
    }

  /** Kept with its 7.0 signature for callers built against it. */
  def doesS3ObjectExist(bucket: String, s3FilePath: String): Boolean =
    getS3ObjectMetadata(bucket, s3FilePath).isDefined

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
    * only be faster or equal, never a new source of download errors. Integrity does not rest on a
    * checksum, since almost no public model publishes one: every range must deliver exactly its
    * own bytes from the object version the HEAD measured (an If-Match pin), so a short, overlong
    * or mixed-version download fails here and falls back instead of passing silently.
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
    val eTag = metadata.flatMap(m => Option(m.getETag))
    val label = s3FilePath.split("/").last
    val connections = effectiveConnections(contentLength)

    // Published for the Python side, which renders progress where this JVM's own output cannot
    // be seen. Scoped to the whole call, including a fallback, so the counters never outlive
    // the transfer they describe.
    if (showProgress) DownloadProgressTracker.begin(label, contentLength)
    try {
      if (connections <= 1) {
        getS3ObjectSequentially(bucket, s3FilePath, tmpFile, contentLength, label, showProgress)
      } else if (!downloadRangesInParallel(
          bucket,
          s3FilePath,
          tmpFile,
          connections,
          contentLength,
          eTag,
          label,
          showProgress)) {
        logger.warn(
          s"Parallel download of $s3FilePath did not complete; " +
            s"falling back to a single connection.")
        // The fallback restarts from zero, so the counters must too or the bar jumps backwards.
        if (showProgress) DownloadProgressTracker.begin(label, contentLength)
        getS3ObjectSequentially(bucket, s3FilePath, tmpFile, contentLength, label, showProgress)
      }
    } finally {
      if (showProgress) DownloadProgressTracker.end()
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

  /** Downloads the object as contiguous ranges into a private part file, then moves it onto
    * `tmpFile`.
    *
    * @return
    *   true only when every range delivered exactly its own bytes from one object version. On
    *   false `tmpFile` is untouched and no worker is still running, so the caller can fall back
    *   to a single connection on the same path.
    */
  private def downloadRangesInParallel(
      bucket: String,
      s3FilePath: String,
      tmpFile: File,
      connections: Int,
      contentLength: Long,
      eTag: Option[String],
      label: String,
      showProgress: Boolean): Boolean = {

    val ranges = AWSGateway.byteRanges(contentLength, connections)
    // byteRanges never returns more ranges than there are bytes, so a tiny object cannot spawn
    // a pool wider than the work available.
    val pool = Executors.newFixedThreadPool(ranges.size, AWSGateway.RangeThreads)
    val progress =
      if (showProgress) Some(new DownloadProgress(contentLength, label)) else None
    val failed = new AtomicBoolean(false)
    val openStreams = ConcurrentHashMap.newKeySet[S3ObjectInputStream]()
    var createdPartFile: Option[File] = None

    // abort() closes the socket under a blocked read, which is what wakes it; an interrupt
    // does not.
    def abortOpenStreams(): Unit = openStreams.asScala.foreach(stream => Try(stream.abort()))

    try {
      // Workers write here, never into tmpFile. A worker that outlives this call can only reach
      // a file nobody reads, so the single-connection fallback owns tmpFile outright. The prefix
      // keeps createTempFile's three-character minimum whatever tmpFile is called.
      val partFile = File.createTempFile(
        tmpFile.getName + ".ranges-",
        ".part",
        tmpFile.getAbsoluteFile.getParentFile)
      createdPartFile = Some(partFile)
      // Preallocate so each worker can write straight at its own offset.
      val preallocate = new RandomAccessFile(partFile, "rw")
      try preallocate.setLength(contentLength)
      finally preallocate.close()

      ranges.foreach { case (start, end) =>
        pool.execute(new Runnable {
          override def run(): Unit =
            try {
              fetchRange(
                bucket,
                s3FilePath,
                eTag,
                start,
                end,
                partFile,
                failed,
                openStreams,
                progress)
            } catch {
              case NonFatal(e) =>
                // Only the first failure is news; the rest are the aborts it caused.
                if (failed.compareAndSet(false, true)) {
                  logger.warn(s"Range $start-$end of $s3FilePath failed: ${e.getMessage}")
                  abortOpenStreams()
                }
            }
        })
      }

      pool.shutdown()
      val finished = pool.awaitTermination(parallelDownloadTimeoutMillis, TimeUnit.MILLISECONDS)
      if (!finished) {
        logger.warn(
          s"Parallel download of $s3FilePath exceeded $parallelDownloadTimeoutMillis ms")
        failed.set(true)
        abortOpenStreams()
      }
      pool.shutdownNow()
      // Never report the outcome while a worker can still write.
      val idle = pool.awaitTermination(AWSGateway.WorkerExitTimeoutSeconds, TimeUnit.SECONDS)
      if (!idle)
        logger.warn(
          s"Range workers for $s3FilePath did not stop within " +
            s"${AWSGateway.WorkerExitTimeoutSeconds} s")
      val complete = finished && idle && !failed.get()
      if (complete) {
        Files.move(partFile.toPath, tmpFile.toPath, StandardCopyOption.REPLACE_EXISTING)
        progress.foreach(_.finish())
      }
      complete
    } catch {
      case NonFatal(e) =>
        logger.warn(s"Parallel download of $s3FilePath failed: ${e.getMessage}")
        false
    } finally {
      // However this call ends -- including an interrupt or a fatal error, which the catch
      // above does not take -- no range may keep streaming behind it.
      failed.set(true)
      abortOpenStreams()
      pool.shutdownNow()
      createdPartFile.foreach(part => if (part.exists() && !part.delete()) part.deleteOnExit())
    }
  }

  /** Writes exactly the bytes of `[start, end]` into `target` at offset `start`.
    *
    * Anything else is a failure, never a partial success: a body that ends early, a body that
    * runs past `end` (a server that ignored the Range header), or a different version of the
    * object (the If-Match pin). Matching the total size across ranges proves nothing, because a
    * short range and a long one can cancel out.
    */
  private def fetchRange(
      bucket: String,
      s3FilePath: String,
      eTag: Option[String],
      start: Long,
      end: Long,
      target: File,
      failed: AtomicBoolean,
      openStreams: java.util.Set[S3ObjectInputStream],
      progress: Option[DownloadProgress]): Unit = {
    val expected = end - start + 1
    // WRITE without CREATE: if the target is already gone, this range has nothing to write to.
    val output = FileChannel.open(target.toPath, StandardOpenOption.WRITE)
    try {
      val request = new GetObjectRequest(bucket, s3FilePath).withRange(start, end)
      // Pins every range to the version the HEAD measured. If the object is replaced
      // mid-download S3 answers 412 and the SDK returns null, rather than the file mixing two
      // versions.
      eTag.foreach(tag => request.withMatchingETagConstraint(tag))
      val s3Object = client.getObject(request)
      if (s3Object == null)
        throw new IllegalStateException(s"$s3FilePath changed during the download")
      val input = s3Object.getObjectContent
      openStreams.add(input)
      var completed = false
      try {
        val buffer =
          new Array[Byte](math.min(AWSGateway.DownloadBufferBytes.toLong, expected).toInt)
        var position = start
        var remaining = expected
        while (remaining > 0) {
          if (failed.get())
            throw new CancellationException(s"another range of $s3FilePath failed")
          val read = input.read(buffer, 0, math.min(buffer.length.toLong, remaining).toInt)
          if (read < 0)
            throw new EOFException(
              s"Range $start-$end of $s3FilePath ended $remaining bytes early")
          val chunk = ByteBuffer.wrap(buffer, 0, read)
          while (chunk.hasRemaining) position += output.write(chunk, position)
          remaining -= read
          progress.foreach(_.advance(read.toLong))
        }
        if (input.read() != -1)
          throw new IOException(
            s"Range $start-$end of $s3FilePath returned more than $expected bytes")
        completed = true
      } finally {
        openStreams.remove(input)
        // close() on a fully read stream returns the connection to the pool; abort() drops it.
        if (completed) input.close() else input.abort()
      }
    } finally output.close()
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

/** Byte counts for the download currently in flight, readable from outside the JVM.
  *
  * A Spark driver launched through py4j writes to a stdout that a notebook front end does not
  * surface, so [[DownloadProgress]]'s own output is invisible there however it is formatted.
  * Python can print to the notebook, but has no view of the transfer -- it is blocked in a py4j
  * call for its whole duration. This is the bridge: the JVM keeps the counters, and the Python
  * side polls them and renders.
  *
  * Every download on the driver shares this one set of counters. Two downloads at once (a job
  * that builds several pipelines on threads) therefore mix their numbers in the display, which is
  * cosmetic: each download's own data is unaffected. Accessors are plain public methods returning
  * primitives, which is what py4j can reach.
  */
object DownloadProgressTracker {

  private val total = new AtomicLong(0L)
  private val done = new AtomicLong(0L)
  @volatile private var name: String = ""
  @volatile private var running: Boolean = false
  private val renderingCallers = new AtomicInteger(0)
  private val generation = new AtomicLong(0L)

  /** Set by Python around a download it draws itself, so this JVM does not draw a second copy
    * onto the same terminal. pyspark starts the JVM with a piped stdin, so the JVM never has a
    * console and would otherwise print one line per percent even in a terminal. Counted rather
    * than a flag: two notebook cells downloading at once each set it, and the JVM stays quiet
    * until both have cleared it.
    */
  def setRenderedByCaller(value: Boolean): Unit =
    if (value) renderingCallers.incrementAndGet()
    else renderingCallers.updateAndGet(callers => math.max(0, callers - 1))

  private[aws] def isRenderedByCaller: Boolean = renderingCallers.get > 0

  /** The configured download_progress, normalised to bar, lines, off or auto. Python resolves
    * auto against its own stdout, which it can see and this JVM cannot.
    */
  def configuredStyle: String = ProgressStyle.configured

  private[aws] def begin(label: String, totalBytes: Long): Unit = {
    name = label
    total.set(totalBytes)
    done.set(0L)
    // After the counters, so a reader that sees the new generation also sees its counters.
    generation.incrementAndGet()
    running = true
  }

  private[aws] def advance(bytes: Long): Unit = done.addAndGet(bytes)

  private[aws] def end(): Unit = running = false

  /** True while a download is in flight. Polled from Python. */
  def isActive: Boolean = running

  /** Name of the object being fetched, for display. Polled from Python. */
  def currentLabel: String = name

  /** Size of the object, or 0 when it is not known. Polled from Python. */
  def bytesTotal: Long = total.get()

  /** Bytes transferred so far. Polled from Python. */
  def bytesDone: Long = done.get()

  /** The whole state in one call, as `active|done|total|label`.
    *
    * Polling the four accessors separately would be four py4j round trips several times a second,
    * on a gateway the caller's own blocked download is already using.
    */
  def snapshot: String = s"${if (running) 1 else 0}|${done.get()}|${total.get()}|$name"

  /** [[snapshot]] behind a number that every new download increments, as
    * `generation|active|done|total|label`. Python notes it when a download call starts, so it can
    * tell that call's download from counters an earlier one left behind (even a download that
    * starts and finishes between two polls) and can follow a second download in one call. A
    * separate method rather than a wider `snapshot`, so older Python keeps parsing that.
    */
  def snapshotWithGeneration: String = s"${generation.get()}|$snapshot"
}

/** How download progress is rendered, and how that is decided. */
private[aws] object ProgressStyle {

  val Bar = "bar"
  val Lines = "lines"
  val Off = "off"
  val Auto = "auto"

  /** Percent step between updates in [[Lines]] mode: one line per whole percent, so a download
    * produces at most 101 lines.
    */
  val LineStepPercent = 1L

  /** Resolved once per JVM: the setting cannot change while it runs, and probing the console on
    * every download would be wasted work.
    */
  lazy val resolved: String = fromConfig(configured)

  private[aws] def configured: String =
    normalise(ConfigLoader.getConfigStringValue(ConfigHelper.pretrainedDownloadProgress))

  private[aws] def normalise(setting: String): String =
    Option(setting).map(_.trim.toLowerCase).getOrElse("") match {
      case explicit @ (Bar | Lines | Off) => explicit
      // "auto", empty, or anything unrecognised: unknown input behaves like the default.
      case _ => Auto
    }

  private[aws] def fromConfig(setting: String, terminal: => Boolean = isTerminal): String =
    normalise(setting) match {
      case Auto => if (terminal) Bar else Lines
      case explicit => explicit
    }

  /** Whether stdout is a real terminal. */
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

/** Byte-accurate progress bar for one download. */
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
    DownloadProgressTracker.advance(bytes)
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
    // Python is drawing this download on its own stdout; a second copy here would interleave.
    if (!DownloadProgressTracker.isRenderedByCaller) style match {
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

  /** Smallest slice worth giving a connection of its own. */
  val MinRangeBytes: Long = 4L * 1024 * 1024

  val DownloadBufferBytes: Int = 256 * 1024

  val MaxParallelConnections: Int = 1000

  val ParallelDownloadTimeoutHours: Long = 12L

  /** How long to wait for range workers to stop after their streams were aborted. An abort makes
    * a blocked read fail at once, so this only bounds a pathological straggler, and even that one
    * can only reach a part file nobody reads.
    */
  val WorkerExitTimeoutSeconds: Long = 30L

  /** Range workers are named daemon threads: one that outlives its download (see
    * `WorkerExitTimeoutSeconds`) must not keep the JVM alive, and a thread dump should say what
    * it is.
    */
  private[aws] val RangeThreads: ThreadFactory = new ThreadFactory {
    private val created = new AtomicInteger(0)

    override def newThread(task: Runnable): Thread = {
      val thread = new Thread(task, s"sparknlp-s3-range-${created.incrementAndGet()}")
      thread.setDaemon(true)
      thread
    }
  }

  /** `download_threads` as a download may use it: at least one connection, and no more than the
    * cap, which also sizes the S3 client's connection pool.
    */
  private[aws] def clampDownloadThreads(requested: Int): Int =
    math.min(MaxParallelConnections, math.max(1, requested))

  /** Connections to open for an object of `contentLength` bytes.
    *
    * `downloadThreads` is an upper bound, not a target: every connection is guaranteed at least
    * `MinRangeBytes`. Without that floor a high setting on a small object opens dozens of
    * connections carrying a few tens of KB each, slower than one connection and billed as one S3
    * GET per range. Anything under two whole ranges takes the single-connection path.
    */
  private[aws] def connectionsFor(contentLength: Long, downloadThreads: Int): Int =
    if (contentLength <= 0) 1
    else math.max(1L, math.min(downloadThreads.toLong, contentLength / MinRangeBytes)).toInt

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
