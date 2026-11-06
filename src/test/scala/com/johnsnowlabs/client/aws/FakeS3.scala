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

package com.johnsnowlabs.client.aws

import com.amazonaws.services.s3.{AbstractAmazonS3, AmazonS3}
import com.amazonaws.services.s3.model.{
  AmazonS3Exception,
  GetObjectRequest,
  ObjectMetadata,
  S3Object,
  S3ObjectInputStream
}

import java.io.{ByteArrayInputStream, File, IOException, InputStream}
import java.nio.file.Files
import java.security.MessageDigest
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger, AtomicLong}
import java.util.concurrent.{ConcurrentHashMap, ConcurrentLinkedQueue, CountDownLatch}
import scala.jdk.CollectionConverters._
import scala.util.Random

/** What the fake sends back for one ranged GET, chosen by the range's start offset. */
sealed trait RangeReply

object RangeReply {

  /** Exactly the requested bytes. */
  case object Exact extends RangeReply

  /** The requested bytes minus the last `missing`, then a clean end of stream. */
  final case class ShortBy(missing: Int) extends RangeReply

  /** The requested bytes plus `extra` bytes past the end, as a server ignoring Range would. */
  final case class LongBy(extra: Int) extends RangeReply

  /** A connection reset on the first read. */
  case object Fail extends RangeReply

  /** Serves `firstBytes`, then blocks like a stalled socket until the stream is aborted. */
  final case class Stall(firstBytes: Int) extends RangeReply
}

/** An in-memory S3 for AWSGateway tests: one object per key, ranged GETs scripted per range, and
  * a record of everything the downloader asked for.
  */
class FakeS3 extends AbstractAmazonS3 {

  /** Bytes and ETag per key. The ETag is computed once, by `put`, so requests inside a test's
    * timed window do not each hash the whole object.
    */
  private val objects = new ConcurrentHashMap[String, (Array[Byte], String)]()
  private val stalls = new ConcurrentLinkedQueue[CountDownLatch]()

  // A typed parameter: `var x: T = _ => ...` would parse as the `= _` default initializer.
  @volatile var replyFor: Long => RangeReply = (_: Long) => RangeReply.Exact

  /** Called with the 1-based count of ranged GETs, before that request is answered. */
  @volatile var onRangeRequest: Int => Unit = (_: Int) => ()

  /** Simulates a store whose HEAD carries no ETag. */
  @volatile var withoutETag: Boolean = false

  val rangeRequests = new ConcurrentLinkedQueue[GetObjectRequest]()
  val sequentialRequests = new AtomicInteger(0)
  val bytesServed = new AtomicLong(0L)
  val openRangeStreams = new AtomicInteger(0)
  @volatile var openRangeStreamsAtFallback: Int = -1
  @volatile var targetBytesAtFallback: Array[Byte] = Array.emptyByteArray

  def put(key: String, data: Array[Byte]): FakeS3 = {
    val md5 = MessageDigest.getInstance("MD5").digest(data).map("%02x".format(_)).mkString
    objects.put(key, (data, md5))
    this
  }

  private def stored(key: String): (Array[Byte], String) =
    Option(objects.get(key)).getOrElse(throw notFound(key))

  def bytes(key: String): Array[Byte] = stored(key)._1

  def eTag(key: String): String = stored(key)._2

  /** Wakes every stalled stream, so a test never leaves threads parked behind it. */
  def releaseStalls(): Unit = stalls.asScala.foreach(_.countDown())

  override def getObjectMetadata(bucketName: String, key: String): ObjectMetadata = {
    val metadata = new ObjectMetadata()
    metadata.setContentLength(bytes(key).length.toLong)
    if (!withoutETag) metadata.setHeader("ETag", eTag(key))
    metadata
  }

  override def getObject(request: GetObjectRequest): S3Object = {
    // Answered one at a time, so a test can replace the object between two requests.
    val answer: Option[(Array[Byte], RangeReply)] = synchronized {
      rangeRequests.add(request)
      onRangeRequest(rangeRequests.size)
      val key = request.getKey
      val pins = request.getMatchingETagConstraints.asScala
      if (pins.nonEmpty && !pins.contains(eTag(key))) None // S3 answered 412
      else Some((bytes(key), replyFor(request.getRange()(0))))
    }
    answer match {
      // The SDK turns a 412 on a matching-ETag constraint into a null object.
      case None => null
      case Some((data, reply)) =>
        val Array(start, end) = request.getRange
        val requested = (end - start + 1).toInt
        val body: InputStream = reply match {
          case RangeReply.Exact => slice(data, start, requested)
          case RangeReply.ShortBy(missing) => slice(data, start, requested - missing)
          case RangeReply.LongBy(extra) => slice(data, start, requested + extra)
          case RangeReply.Fail =>
            new InputStream {
              override def read(): Int = throw new IOException("Connection reset")
            }
          case RangeReply.Stall(firstBytes) => new StallingStream(slice(data, start, firstBytes))
        }
        openRangeStreams.incrementAndGet()
        val s3Object = new S3Object()
        s3Object.setKey(request.getKey)
        // No metrics wrapper, so S3ObjectInputStream.abort() closes the tracked stream directly.
        s3Object.setObjectContent(new S3ObjectInputStream(new TrackedStream(body), null, false))
        s3Object
    }
  }

  /** The single-connection path. Records what the ranged attempt left behind when it starts. */
  override def getObject(request: GetObjectRequest, destinationFile: File): ObjectMetadata = {
    openRangeStreamsAtFallback = openRangeStreams.get()
    targetBytesAtFallback =
      if (destinationFile.exists()) Files.readAllBytes(destinationFile.toPath)
      else Array.emptyByteArray
    sequentialRequests.incrementAndGet()
    Files.write(destinationFile.toPath, bytes(request.getKey))
    getObjectMetadata(request.getBucketName, request.getKey)
  }

  private def slice(data: Array[Byte], start: Long, length: Int): InputStream = {
    val from = start.toInt
    new ByteArrayInputStream(data, from, math.max(0, math.min(length, data.length - from)))
  }

  private def notFound(key: String): AmazonS3Exception = {
    val error = new AmazonS3Exception(s"Not Found: $key")
    error.setStatusCode(404)
    error
  }

  /** Counts served bytes and open streams. S3ObjectInputStream.abort() closes this stream. */
  private class TrackedStream(in: InputStream) extends InputStream {
    private val closed = new AtomicBoolean(false)

    override def read(): Int = {
      val b = in.read()
      if (b >= 0) bytesServed.incrementAndGet()
      b
    }

    override def read(buffer: Array[Byte], offset: Int, length: Int): Int = {
      val n = in.read(buffer, offset, length)
      if (n > 0) bytesServed.addAndGet(n.toLong)
      n
    }

    override def close(): Unit =
      if (closed.compareAndSet(false, true)) {
        openRangeStreams.decrementAndGet()
        in.close()
      }
  }

  /** Serves its head, then blocks until closed. Like a socket read, it ignores interrupts. */
  private class StallingStream(head: InputStream) extends InputStream {
    private val released = new CountDownLatch(1)
    stalls.add(released)

    override def read(): Int = {
      val one = new Array[Byte](1)
      if (read(one, 0, 1) < 0) -1 else one(0) & 0xff
    }

    override def read(buffer: Array[Byte], offset: Int, length: Int): Int = {
      val n = head.read(buffer, offset, length)
      if (n >= 0) n
      else {
        awaitReleaseIgnoringInterrupts()
        throw new IOException("Stream aborted")
      }
    }

    override def close(): Unit = released.countDown()

    private def awaitReleaseIgnoringInterrupts(): Unit = {
      var interrupted = false
      while (released.getCount > 0) {
        try released.await()
        catch { case _: InterruptedException => interrupted = true }
      }
      if (interrupted) Thread.currentThread().interrupt()
    }
  }
}

object FakeS3 {

  def payload(size: Int, seed: Long = 42L): Array[Byte] = {
    val data = new Array[Byte](size)
    new Random(seed).nextBytes(data)
    data
  }

  /** A gateway that talks to `s3`, splits downloads over up to `threads` connections, and gives
    * up on a ranged attempt after `timeoutMillis`.
    */
  def gatewayFor(s3: FakeS3, threads: Int, timeoutMillis: Long = 60000L): AWSGateway =
    new AWSGateway(
      accessKeyId = "",
      secretAccessKey = "",
      sessionToken = "",
      awsProfile = "",
      region = "us-east-1",
      credentialsType = "public") {
      override lazy val client: AmazonS3 = s3
      override protected lazy val downloadThreads: Int = threads
      override protected def parallelDownloadTimeoutMillis: Long = timeoutMillis
    }
}
