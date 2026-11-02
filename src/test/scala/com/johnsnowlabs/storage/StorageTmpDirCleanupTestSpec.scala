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

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.commons.io.FileUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path, RawLocalFileSystem}
import org.apache.spark.scheduler.SparkListenerApplicationEnd
import org.scalatest.BeforeAndAfterEach
import org.scalatest.flatspec.AnyFlatSpec

import java.io.{File, IOException}
import java.net.URI
import java.nio.file.{AccessDeniedException, Files}

/** Answers for s3a://test-bucket/<absolute local path> from the local disk. */
class FakeS3FileSystem extends RawLocalFileSystem {
  override def getUri: URI = URI.create("s3a://test-bucket/")
  override def getScheme: String = "s3a"
}

/** What S3A raises when the IAM role lacks s3:DeleteObject. */
class DenyingS3FileSystem extends FakeS3FileSystem {
  override def delete(p: Path, recursive: Boolean): Boolean =
    throw new AccessDeniedException(p.toString, null, "s3a delete: 403 Forbidden")
}

/** How EMRFS reports the same 403: an IOException carrying the AWS SDK message. */
class EmrfsDenyingS3FileSystem extends FakeS3FileSystem {
  override def delete(p: Path, recursive: Boolean): Boolean =
    throw new IOException(
      "Delete failed",
      new RuntimeException(
        "Access Denied (Service: Amazon S3; Status Code: 403; Error Code: AccessDenied)"))
}

/** Reports a failed delete by returning false instead of throwing. */
class NonDeletingS3FileSystem extends FakeS3FileSystem {
  override def delete(p: Path, recursive: Boolean): Boolean = false
}

class StorageTmpDirCleanupTestSpec extends AnyFlatSpec with BeforeAndAfterEach {

  private var localRoot: File = _

  override def beforeEach(): Unit = {
    localRoot = Files.createTempDirectory("sparknlp_1454_").toFile
    StorageTmpDirCleanup.cleanup(fakeS3Conf(classOf[FakeS3FileSystem]))
  }

  override def afterEach(): Unit = {
    StorageTmpDirCleanup.cleanup(fakeS3Conf(classOf[FakeS3FileSystem]))
    FileUtils.deleteDirectory(localRoot)
  }

  private def fakeS3Conf(fs: Class[_ <: FakeS3FileSystem]): Configuration = {
    val conf = new Configuration()
    conf.set("fs.s3a.impl", fs.getName)
    conf.setBoolean("fs.s3a.impl.disable.cache", true)
    conf
  }

  private def s3Path(relative: String): Path =
    new Path(s"s3a://test-bucket${localRoot.getAbsolutePath}/$relative")

  private def writeFile(relative: String): File = {
    val file = new File(localRoot, relative)
    file.getParentFile.mkdirs()
    Files.write(file.toPath, "data".getBytes)
    file
  }

  "StorageTmpDirCleanup" should "delete a registered S3 _cdx folder and nothing above it" taggedAs FastTest in {
    writeFile("base/keep.txt")
    writeFile("base/abc123_cdx/EMBEDDINGS_glove_100d/000001.sst")

    StorageTmpDirCleanup.register(s3Path("base/abc123_cdx"), ResourceHelper.spark.sparkContext)
    val leftBehind = StorageTmpDirCleanup.cleanup(fakeS3Conf(classOf[FakeS3FileSystem]))

    assert(leftBehind.isEmpty)
    assert(!new File(localRoot, "base/abc123_cdx").exists())
    assert(new File(localRoot, "base/keep.txt").exists())
  }

  it should "leave another application's _cdx folder under the same base alone" taggedAs FastTest in {
    writeFile("base/abc123_cdx/EMBEDDINGS_glove_100d/000001.sst")
    writeFile("base/other9_cdx/EMBEDDINGS_glove_100d/000001.sst")

    StorageTmpDirCleanup.register(s3Path("base/abc123_cdx"), ResourceHelper.spark.sparkContext)
    StorageTmpDirCleanup.cleanup(fakeS3Conf(classOf[FakeS3FileSystem]))

    assert(!new File(localRoot, "base/abc123_cdx").exists())
    assert(new File(localRoot, "base/other9_cdx/EMBEDDINGS_glove_100d/000001.sst").exists())
  }

  it should "keep going when S3A denies the delete" taggedAs FastTest in {
    writeFile("base/abc123_cdx/EMBEDDINGS_glove_100d/000001.sst")

    StorageTmpDirCleanup.register(s3Path("base/abc123_cdx"), ResourceHelper.spark.sparkContext)
    val leftBehind = StorageTmpDirCleanup.cleanup(fakeS3Conf(classOf[DenyingS3FileSystem]))

    assert(leftBehind == Seq(s3Path("base/abc123_cdx")))
    assert(new File(localRoot, "base/abc123_cdx/EMBEDDINGS_glove_100d/000001.sst").exists())
    assert(StorageTmpDirCleanup.pending.isEmpty)
  }

  it should "keep going when EMRFS denies the delete" taggedAs FastTest in {
    writeFile("base/abc123_cdx/EMBEDDINGS_glove_100d/000001.sst")

    StorageTmpDirCleanup.register(s3Path("base/abc123_cdx"), ResourceHelper.spark.sparkContext)
    val leftBehind = StorageTmpDirCleanup.cleanup(fakeS3Conf(classOf[EmrfsDenyingS3FileSystem]))

    assert(leftBehind == Seq(s3Path("base/abc123_cdx")))
    assert(new File(localRoot, "base/abc123_cdx/EMBEDDINGS_glove_100d/000001.sst").exists())
    assert(StorageTmpDirCleanup.pending.isEmpty)
  }

  it should "report a folder the filesystem could not delete without an exception" taggedAs FastTest in {
    writeFile("base/abc123_cdx/EMBEDDINGS_glove_100d/000001.sst")

    StorageTmpDirCleanup.register(s3Path("base/abc123_cdx"), ResourceHelper.spark.sparkContext)
    val leftBehind = StorageTmpDirCleanup.cleanup(fakeS3Conf(classOf[NonDeletingS3FileSystem]))

    assert(leftBehind == Seq(s3Path("base/abc123_cdx")))
    assert(StorageTmpDirCleanup.pending.isEmpty)
  }

  it should "recognise S3A and EMRFS permission errors, and nothing else" taggedAs FastTest in {
    val path = s3Path("base/abc123_cdx")
    val s3a = intercept[AccessDeniedException] {
      new DenyingS3FileSystem().delete(path, true)
    }
    val emrfs = intercept[IOException] {
      new EmrfsDenyingS3FileSystem().delete(path, true)
    }

    assert(StorageTmpDirCleanup.isPermissionDenied(s3a))
    assert(StorageTmpDirCleanup.isPermissionDenied(emrfs))
    assert(!StorageTmpDirCleanup.isPermissionDenied(new IOException("Connection reset")))
  }

  it should "not fail when the registered folder was never written" taggedAs FastTest in {
    StorageTmpDirCleanup.register(s3Path("base/never_cdx"), ResourceHelper.spark.sparkContext)
    val leftBehind = StorageTmpDirCleanup.cleanup(fakeS3Conf(classOf[FakeS3FileSystem]))

    assert(leftBehind.isEmpty)
    assert(StorageTmpDirCleanup.pending.isEmpty)
  }

  it should "clean up when Spark reports the end of the application" taggedAs FastTest in {
    writeFile("base/abc123_cdx/EMBEDDINGS_glove_100d/000001.sst")
    StorageTmpDirCleanup.register(s3Path("base/abc123_cdx"), ResourceHelper.spark.sparkContext)

    new StorageTmpDirCleanup.CleanupListener(fakeS3Conf(classOf[FakeS3FileSystem]))
      .onApplicationEnd(SparkListenerApplicationEnd(System.currentTimeMillis()))

    assert(!new File(localRoot, "base/abc123_cdx").exists())
    assert(StorageTmpDirCleanup.pending.isEmpty)
  }

  "StorageLocator" should "register its S3 temporary folder for cleanup" taggedAs FastTest in {
    val cdx = s3Path("base/abc123_cdx").toString

    new StorageLocator("EMBEDDINGS", "glove_100d", ResourceHelper.spark) {
      override protected def getTmpLocation: String = cdx
    }

    assert(StorageTmpDirCleanup.pending == Seq(new Path(cdx)))
  }

  it should "pick the S3 folder to clean up from the configured path and the default filesystem" taggedAs FastTest in {
    val local = FileSystem.getLocal(new Configuration())
    val s3Default = new FakeS3FileSystem()

    assert(
      StorageLocator
        .s3TmpFolder("s3a://bucket/base/abc123_cdx", local)
        .contains(new Path("s3a://bucket/base/abc123_cdx")))
    assert(
      StorageLocator
        .s3TmpFolder("s3://bucket/base/abc123_cdx", local)
        .contains(new Path("s3://bucket/base/abc123_cdx")))
    assert(
      StorageLocator
        .s3TmpFolder("/tmp/hadoop/abc123_cdx", s3Default)
        .contains(new Path("s3a://test-bucket/tmp/hadoop/abc123_cdx")))
    assert(StorageLocator.s3TmpFolder("/tmp/hadoop/abc123_cdx", local).isEmpty)
    assert(StorageLocator.s3TmpFolder("hdfs://namenode/tmp/abc123_cdx", s3Default).isEmpty)
  }

  "ConfigLoader" should "put cluster_tmp_dir in a per-JVM _cdx subfolder of the configured path" taggedAs FastTest in {
    ResourceHelper.spark
    val tmpLocation = ConfigLoader.getConfigStringValue(ConfigHelper.storageTmpDir)

    assert(tmpLocation.matches(".+/[0-9a-f]{12}_cdx"), tmpLocation)
  }

}
