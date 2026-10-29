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

package com.johnsnowlabs.nlp.pretrained

import com.amazonaws.services.s3.model.ObjectMetadata
import com.johnsnowlabs.client.aws.AWSGateway
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.Version
import org.apache.hadoop.fs.Path
import org.scalatest.flatspec.AnyFlatSpec

import java.io.{ByteArrayOutputStream, File, IOException}
import java.nio.file.Files
import java.security.MessageDigest
import java.sql.Timestamp
import java.util.UUID
import java.util.concurrent.{ConcurrentLinkedQueue, CyclicBarrier, TimeUnit}
import java.util.zip.{ZipEntry, ZipOutputStream}
import scala.jdk.CollectionConverters._
import scala.util.{Failure, Random, Try}

class CacheInstallTest extends AnyFlatSpec {

  private def zipOf(entries: (String, Array[Byte])*): Array[Byte] = {
    val bytes = new ByteArrayOutputStream()
    val zip = new ZipOutputStream(bytes)
    entries.foreach { case (name, data) =>
      zip.putNextEntry(new ZipEntry(name))
      zip.write(data)
      zip.closeEntry()
    }
    zip.close()
    bytes.toByteArray
  }

  private def randomBytes(size: Int, seed: Long): Array[Byte] = {
    val data = new Array[Byte](size)
    new Random(seed).nextBytes(data)
    data
  }

  private def md5(data: Array[Byte]): String =
    MessageDigest.getInstance("MD5").digest(data).map("%02X".format(_)).mkString

  private def resource(checksum: String): ResourceMetadata =
    ResourceMetadata(
      name = "fake_model",
      language = Some("en"),
      libVersion = Some(Version(7, 0, 0)),
      sparkVersion = Some(Version(3, 5)),
      readyToUse = true,
      time = new Timestamp(1700000000000L),
      isZipped = true,
      checksum = checksum)

  /** A downloader whose S3 hands back whatever `serve()` returns, then runs `afterServe`; each
    * download's `showProgress` lands in `progressRequests`.
    */
  private def downloader(
      cache: File,
      serve: () => Array[Byte],
      afterServe: () => Unit = () => (),
      progressRequests: ConcurrentLinkedQueue[Boolean] = new ConcurrentLinkedQueue[Boolean]())
      : S3ResourceDownloader =
    new S3ResourceDownloader("bucket", "public/models", cache.getAbsolutePath, "public") {
      override lazy val awsGateway: AWSGateway =
        new AWSGateway(
          accessKeyId = "",
          secretAccessKey = "",
          sessionToken = "",
          awsProfile = "",
          region = "us-east-1",
          credentialsType = "public") {
          override def getS3Object(
              bucket: String,
              s3FilePath: String,
              tmpFile: File,
              objectMetadata: Option[ObjectMetadata],
              showProgress: Boolean): Unit = {
            progressRequests.add(showProgress)
            Files.write(tmpFile.toPath, serve())
            afterServe()
          }
        }
    }

  /** Cache entries a later lookup could find; Hadoop's hidden .crc files are excluded. */
  private def visible(cache: File): Seq[String] =
    Option(cache.listFiles()).toSeq.flatten.map(_.getName).filterNot(_.startsWith(".")).sorted

  private def install(
      rd: S3ResourceDownloader,
      cache: File,
      res: ResourceMetadata): Option[String] =
    rd.downloadAndUnzipFile(
      new Path(cache.getAbsolutePath, res.fileName),
      res,
      s"public/models/${res.fileName}",
      None)

  "Installing a model" should "reject a checksum mismatch before anything reaches the cache" taggedAs FastTest in {
    val cache = Files.createTempDirectory("cache-install").toFile
    val rd = downloader(cache, () => zipOf("a.txt" -> "hello".getBytes("UTF-8")))
    intercept[IllegalArgumentException](install(rd, cache, resource(checksum = "0" * 32)))
    assert(visible(cache).isEmpty)
  }

  it should "accept a matching checksum and leave only the model directory" taggedAs FastTest in {
    val cache = Files.createTempDirectory("cache-install").toFile
    val archive = zipOf("a.txt" -> "hello".getBytes("UTF-8"))
    val res = resource(checksum = md5(archive))
    val installed = install(downloader(cache, () => archive), cache, res)

    assert(visible(cache) == Seq(res.fileName.stripSuffix(".zip")))
    assert(
      new String(Files.readAllBytes(new File(installed.get, "a.txt").toPath), "UTF-8") == "hello")
  }

  it should "leave neither a partial model nor the bad zip when the archive is truncated" taggedAs FastTest in {
    val cache = Files.createTempDirectory("cache-install").toFile
    val archive = zipOf("a.bin" -> randomBytes(200000, 1L), "b.bin" -> randomBytes(200000, 2L))
    var served = archive.take(archive.length * 3 / 4)
    val rd = downloader(cache, () => served)
    val res = resource(checksum = "") // public models publish no checksum

    intercept[IOException](install(rd, cache, res))
    assert(visible(cache).isEmpty, s"left behind: ${visible(cache)}")

    served = archive
    val installed = install(rd, cache, res)
    assert(
      new File(installed.get, "b.bin").length == 200000L,
      "a retry must download again, not reuse half a model")
  }

  it should "reclaim a part file left by a run killed mid-copy but leave a live one alone" taggedAs FastTest in {
    val cache = Files.createTempDirectory("cache-install").toFile
    val res = resource(checksum = "")
    val stale = new File(cache, s"${res.fileName}.part-${UUID.randomUUID()}")
    val live = new File(cache, s"${res.fileName}.part-${UUID.randomUUID()}")
    Files.write(stale.toPath, "half a zip".getBytes("UTF-8"))
    Files.write(live.toPath, "another installer is still copying".getBytes("UTF-8"))
    assert(stale.setLastModified(System.currentTimeMillis() - TimeUnit.HOURS.toMillis(2)))
    val installed =
      install(downloader(cache, () => zipOf("a.txt" -> "hello".getBytes("UTF-8"))), cache, res)

    assert(new File(installed.get, "a.txt").exists())
    assert(!stale.exists(), "a part untouched for hours belongs to a run that died")
    assert(live.exists(), "a fresh part may belong to an installer that is still copying")
  }

  it should "let two installers of the same model both succeed" taggedAs FastTest in {
    val MiB = 1024 * 1024
    for (round <- 1 to 3) {
      val cache = Files.createTempDirectory("cache-install").toFile
      val archive =
        zipOf("a.bin" -> randomBytes(4 * MiB, 1L), "b.bin" -> randomBytes(4 * MiB, 2L))
      // Both installers finish downloading at the same instant, so their moves into the cache
      // and their unzips overlap.
      val bothDownloaded = new CyclicBarrier(2)
      val rd =
        downloader(cache, () => archive, () => { bothDownloaded.await(30, TimeUnit.SECONDS); () })
      val res = resource(checksum = "")
      val results = new ConcurrentLinkedQueue[Try[Option[String]]]()
      val installers = (1 to 2).map { _ =>
        new Thread(() => { results.add(Try(install(rd, cache, res))); () })
      }
      installers.foreach(_.start())
      installers.foreach(_.join(60000L))

      val failures = results.asScala.collect { case Failure(e) => e.toString }.toList
      assert(results.size == 2 && failures.isEmpty, s"round $round: $failures")
      assert(results.asScala.map(_.get).toSet.size == 1, s"round $round: different paths")
      val model = new File(results.peek().get.get)
      assert(new File(model, "a.bin").length == 4L * MiB, s"round $round")
      assert(new File(model, "b.bin").length == 4L * MiB, s"round $round")
      assert(visible(cache) == Seq(model.getName), s"round $round left: ${visible(cache)}")
    }
  }

  "downloadAndUnzipFile(destination, resource, s3FilePath)" should "keep its 7.0 signature" taggedAs FastTest in {
    val cache = Files.createTempDirectory("cache-install").toFile
    val res = resource(checksum = "")
    val installed: Option[String] =
      downloader(cache, () => zipOf("a.txt" -> "hello".getBytes("UTF-8")))
        .downloadAndUnzipFile(
          new Path(cache.getAbsolutePath, res.fileName),
          res,
          s"public/models/${res.fileName}")

    assert(new File(installed.get, "a.txt").exists())
  }

  "downloadAndUnzipFile(s3FilePath, unzip)" should "ask for progress, which downloadModelDirectly shows" taggedAs FastTest in {
    val cache = Files.createTempDirectory("cache-install").toFile
    val res = resource(checksum = "")
    val progressRequests = new ConcurrentLinkedQueue[Boolean]()
    val rd = downloader(
      cache,
      () => zipOf("a.txt" -> "hello".getBytes("UTF-8")),
      progressRequests = progressRequests)
    rd.downloadAndUnzipFile(s"public/models/${res.fileName}", unzip = true)

    assert(progressRequests.asScala.toList == List(true))
  }
}
