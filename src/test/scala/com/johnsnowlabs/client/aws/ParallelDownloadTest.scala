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

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File
import java.nio.file.Files
import java.util.concurrent.atomic.AtomicBoolean
import scala.jdk.CollectionConverters._

class ParallelDownloadTest extends AnyFlatSpec {

  private val MiB = 1024 * 1024
  private val Key = "public/models/fake_model_en_7.0.0_3.0_1.zip"

  private def waitUntil(condition: => Boolean, millis: Long): Boolean = {
    val deadline = System.nanoTime() + millis * 1000000L
    while (!condition && System.nanoTime() < deadline) Thread.sleep(10)
    condition
  }

  /** Runs a download on its own thread, swallowing whatever it throws. */
  private def downloadInBackground(gateway: AWSGateway, file: File): Thread = {
    val thread = new Thread(() => {
      try gateway.getS3Object("bucket", Key, file, showProgress = false)
      catch { case _: Throwable => () }
    })
    thread.start()
    thread
  }

  private def target(): File =
    File.createTempFile("model", ".zip", Files.createTempDirectory("ranged-download").toFile)

  private def contentOf(file: File): Array[Byte] = Files.readAllBytes(file.toPath)

  "A ranged download" should "assemble the object byte for byte from ranges pinned to its ETag" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(40 * MiB + 3))
    val file = target()
    FakeS3.gatewayFor(s3, threads = 10).getS3Object("bucket", Key, file, showProgress = false)

    assert(contentOf(file) sameElements s3.bytes(Key))
    assert(s3.rangeRequests.size == 10)
    assert(
      s3.rangeRequests.asScala.forall(_.getMatchingETagConstraints.asScala == Seq(s3.eTag(Key))))
    assert(s3.sequentialRequests.get == 0)
  }

  it should "use one plain GET when download_threads is 1" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(40 * MiB))
    val file = target()
    FakeS3.gatewayFor(s3, threads = 1).getS3Object("bucket", Key, file, showProgress = false)

    assert(s3.rangeRequests.isEmpty)
    assert(s3.sequentialRequests.get == 1)
    assert(contentOf(file) sameElements s3.bytes(Key))
  }

  it should "use one plain GET for an object under 8 MiB" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(8 * MiB - 1))
    val file = target()
    FakeS3.gatewayFor(s3, threads = 10).getS3Object("bucket", Key, file, showProgress = false)

    assert(s3.rangeRequests.isEmpty)
    assert(s3.sequentialRequests.get == 1)
    assert(contentOf(file) sameElements s3.bytes(Key))
  }

  it should "reject a short range even when another range makes up the total" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(40 * MiB))
    val ranges = AWSGateway.byteRanges(40L * MiB, 10)
    s3.replyFor = start =>
      if (start == ranges(0)._1) RangeReply.ShortBy(10)
      else if (start == ranges(1)._1) RangeReply.LongBy(10)
      else RangeReply.Exact
    val file = target()
    FakeS3.gatewayFor(s3, threads = 10).getS3Object("bucket", Key, file, showProgress = false)

    assert(
      s3.sequentialRequests.get == 1,
      "mismatched ranges must send the download to the fallback")
    assert(contentOf(file) sameElements s3.bytes(Key))
  }

  it should "reject a range body that runs past the requested end" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB))
    val ranges = AWSGateway.byteRanges(16L * MiB, 4)
    s3.replyFor = start => if (start == ranges(1)._1) RangeReply.LongBy(1) else RangeReply.Exact
    val file = target()
    FakeS3.gatewayFor(s3, threads = 4).getS3Object("bucket", Key, file, showProgress = false)

    assert(s3.sequentialRequests.get == 1)
    assert(contentOf(file) sameElements s3.bytes(Key))
  }

  it should "not stitch an object replaced mid-download from two versions" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB, seed = 1L))
    val replacement = FakeS3.payload(16 * MiB, seed = 2L)
    s3.onRangeRequest = n => if (n == 2) s3.put(Key, replacement)
    val file = target()
    FakeS3.gatewayFor(s3, threads = 4).getS3Object("bucket", Key, file, showProgress = false)

    assert(s3.sequentialRequests.get == 1)
    assert(contentOf(file) sameElements replacement)
  }

  it should "still download, unpinned, when the HEAD carries no ETag" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB))
    s3.withoutETag = true
    val file = target()
    FakeS3.gatewayFor(s3, threads = 4).getS3Object("bucket", Key, file, showProgress = false)

    assert(s3.rangeRequests.asScala.forall(_.getMatchingETagConstraints.isEmpty))
    assert(s3.sequentialRequests.get == 0)
    assert(contentOf(file) sameElements s3.bytes(Key))
  }

  it should "abort the other ranges as soon as one fails, then fall back once" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(40 * MiB))
    val failing = AWSGateway.byteRanges(40L * MiB, 10)(3)._1
    s3.replyFor =
      start => if (start == failing) RangeReply.Fail else RangeReply.Stall(firstBytes = 64 * 1024)
    val file = target()
    try {
      val started = System.nanoTime()
      FakeS3
        .gatewayFor(s3, threads = 10, timeoutMillis = 5000L)
        .getS3Object("bucket", Key, file, showProgress = false)
      val seconds = (System.nanoTime() - started) / 1e9

      assert(seconds < 4.0, s"took $seconds s: the failure must cancel the stalled ranges")
      assert(s3.openRangeStreamsAtFallback == 0, "every range stream must be closed first")
      assert(s3.bytesServed.get < 40L * MiB, "the other ranges must not run to completion")
      assert(s3.sequentialRequests.get == 1)
      assert(contentOf(file) sameElements s3.bytes(Key))
    } finally s3.releaseStalls()
  }

  it should "abort a range stalled past the timeout and leave the target untouched" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB))
    s3.replyFor =
      start => if (start == 0L) RangeReply.Stall(firstBytes = 64 * 1024) else RangeReply.Exact
    val file = target()
    Files.write(file.toPath, "untouched".getBytes("UTF-8"))
    try {
      FakeS3
        .gatewayFor(s3, threads = 4, timeoutMillis = 300L)
        .getS3Object("bucket", Key, file, showProgress = false)

      assert(s3.openRangeStreamsAtFallback == 0)
      assert(
        new String(s3.targetBytesAtFallback, "UTF-8") == "untouched",
        "ranges must never write the file the fallback writes")
      assert(contentOf(file) sameElements s3.bytes(Key))
      assert(file.getParentFile.listFiles().forall(!_.getName.endsWith(".part")))
    } finally s3.releaseStalls()
  }

  it should "keep two concurrent ranged downloads byte-exact and apart" taggedAs FastTest in {
    val s3 = new FakeS3()
      .put("a.zip", FakeS3.payload(16 * MiB, seed = 1L))
      .put("b.zip", FakeS3.payload(16 * MiB, seed = 2L))
    val gateway = FakeS3.gatewayFor(s3, threads = 4)
    val files = Map("a.zip" -> target(), "b.zip" -> target())
    val downloads = files.toSeq.map { case (key, file) =>
      new Thread(() => gateway.getS3Object("bucket", key, file, showProgress = false))
    }
    downloads.foreach(_.start())
    downloads.foreach(_.join(60000L))

    files.foreach { case (key, file) => assert(contentOf(file) sameElements s3.bytes(key), key) }
    assert(s3.sequentialRequests.get == 0)
  }

  it should "leave the progress tracker idle when an unshown download falls back" taggedAs FastTest in {
    DownloadProgressTracker.end()
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB))
    s3.replyFor = start => if (start == 0L) RangeReply.Fail else RangeReply.Exact
    FakeS3.gatewayFor(s3, threads = 4).getS3Object("bucket", Key, target(), showProgress = false)

    assert(s3.sequentialRequests.get == 1)
    assert(
      !DownloadProgressTracker.isActive,
      "an index fetch must never look like a user download")
  }

  it should "keep two downloads of one object into one folder apart when one of them fails" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB))
    val failing = AWSGateway.byteRanges(16L * MiB, 4)(2)._1
    val failedOnce = new AtomicBoolean(false)
    s3.replyFor = start =>
      if (start == failing && failedOnce.compareAndSet(false, true)) RangeReply.Fail
      else RangeReply.Exact
    val folder = Files.createTempDirectory("ranged-download").toFile
    val files = Seq(1, 2).map(_ => File.createTempFile("model", ".zip", folder))
    val gateway = FakeS3.gatewayFor(s3, threads = 4)
    files.map(file => downloadInBackground(gateway, file)).foreach(_.join(60000L))

    files.foreach(file => assert(contentOf(file) sameElements s3.bytes(Key), file.getName))
    assert(s3.sequentialRequests.get == 1, "only the download whose range failed may fall back")
    assert(folder.listFiles().forall(!_.getName.endsWith(".part")))
  }

  it should "fall back to one connection when its part file cannot be created" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB))
    val file = target()
    val folder = file.getParentFile
    assume(folder.setWritable(false), "needs a folder this user can make read-only")
    try {
      FakeS3.gatewayFor(s3, threads = 4).getS3Object("bucket", Key, file, showProgress = false)

      assert(s3.sequentialRequests.get == 1)
      assert(contentOf(file) sameElements s3.bytes(Key))
    } finally folder.setWritable(true)
  }

  it should "abort its streams when the calling thread is interrupted" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB))
    s3.replyFor = _ => RangeReply.Stall(firstBytes = 64 * 1024)
    try {
      val caller = downloadInBackground(FakeS3.gatewayFor(s3, threads = 4), target())
      assert(waitUntil(s3.openRangeStreams.get == 4, 10000L), "all four ranges should be stalled")
      caller.interrupt()
      caller.join(10000L)

      assert(
        waitUntil(s3.openRangeStreams.get == 0, 5000L),
        s"${s3.openRangeStreams.get} ranges still streaming after the caller gave up")
    } finally s3.releaseStalls()
  }

  it should "run its ranges on named daemon threads" taggedAs FastTest in {
    val s3 = new FakeS3().put(Key, FakeS3.payload(16 * MiB))
    s3.replyFor = _ => RangeReply.Stall(firstBytes = 64 * 1024)
    val caller = downloadInBackground(FakeS3.gatewayFor(s3, threads = 4), target())
    try {
      assert(waitUntil(s3.openRangeStreams.get == 4, 10000L), "all four ranges should be stalled")
      val workers = Thread.getAllStackTraces.keySet.asScala
        .filter(_.getName.startsWith("sparknlp-s3-range-"))
      assert(workers.size >= 4, s"range threads: ${workers.map(_.getName)}")
      assert(workers.forall(_.isDaemon), "a straggler must not keep the JVM alive")
    } finally {
      s3.releaseStalls()
      caller.join(10000L)
    }
  }
}
