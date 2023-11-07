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

package com.johnsnowlabs.util

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers._
import java.nio.file.{Files, Paths, Path}
import java.io.File
import com.johnsnowlabs.util.FileHelper
import org.scalatest.BeforeAndAfter
import java.util.UUID

class ZipArchiveUtilTestSpec extends AnyFlatSpec with BeforeAndAfter {
  private val tmpDirPath: String = UUID.randomUUID().toString.takeRight(12) + "_onnx"
  var tmpFolder: String = _

  before {
    // create temp dir for testing
    tmpFolder = Files
      .createDirectory(Paths.get(tmpDirPath))
      .toAbsolutePath
      .toString

    // create files and dirs for recusive testing
    new File(tmpFolder, "fileA").createNewFile()
    Files.createDirectory(Paths.get(tmpDirPath, "dir"))
    Files.createFile(Paths.get(tmpFolder, "dir", "fileA"))
    Files.createFile(Paths.get(tmpFolder, "dir", "fileB"))
    Files.createDirectory(Paths.get(tmpDirPath, "dir", "dir2"))
    Files.createFile(Paths.get(tmpDirPath, "dir", "dir2", "fileC"))
  }

  after {
    // delete the temp directory
    FileHelper.delete(tmpFolder)
  }

  "listFilesRecursive" should "throw exception if the file doesn't exist" taggedAs FastTest in {
    val isIOException =
      try {
        ZipArchiveUtil.listFilesRecursive(new File("a"))
        false
      } catch {
        case e: java.io.IOException => true
        case _: Throwable => false
      }
    assert(isIOException)
  }

  "listFilesRecursive" should "return a single item list if give a file" taggedAs FastTest in {
    val list = ZipArchiveUtil.listFilesRecursive(new File(tmpFolder, "fileA"))
    assert(list.length == 1)
    assert(list.head.equals(new File(tmpFolder, "fileA")))
  }

  "listFilesRecursive" should "return a single item list if give a file within folder" taggedAs FastTest in {
    val list = ZipArchiveUtil.listFilesRecursive(new File(tmpFolder, "dir/fileA"))
    assert(list.length == 1)
    assert(list.head.equals(new File(tmpFolder, "dir/fileA")))
  }

  "listFilesRecursive" should "return a list with 3 items if give the dir folder" taggedAs FastTest in {
    val list = ZipArchiveUtil.listFilesRecursive(new File(tmpFolder, "dir"))
    assert(list.length == 3)

    list.toSet should contain theSameElementsAs Set(
      new File(tmpFolder, "dir/dir2/fileC"),
      new File(tmpFolder, "dir/fileA"),
      new File(tmpFolder, "dir/fileB"))
  }

  "addFileToZipEntry" should "return zip entry with absolute setting" taggedAs FastTest in {
    val zipEntry = ZipArchiveUtil.addFileToZipEntry(new File("fileA"), null, false)
    assert(zipEntry.getName == "fileA")
  }

  "addFileToZipEntry" should "return zip entry with relative setting" taggedAs FastTest in {
    val zipEntry = ZipArchiveUtil.addFileToZipEntry(new File("dir/fileA"), new File("dir"), true)
    assert(zipEntry.getName == "fileA")
  }

  "addFileToZipEntry" should "return zip entry full path with absolute setting" taggedAs FastTest in {
    val zipEntry = ZipArchiveUtil.addFileToZipEntry(new File("dir/fileA"), new File("dir"), false)
    assert(zipEntry.getName == "fileA")
  }

  "createZip" should "create zip for a single file" taggedAs FastTest in {
    ZipArchiveUtil.createZip(
      List(new File(tmpFolder, "dir/fileA")),
      new File(tmpFolder, "targetA.zip"),
      null)

    assert(new File(tmpFolder, "targetA.zip").exists())
  }

  "createZip" should "create zip" taggedAs FastTest in {
    ZipArchiveUtil.createZip(
      List(
        new File(Paths.get(tmpFolder, "dir", "fileA").toString),
        new File(Paths.get(tmpFolder, "dir", "fileB").toString)),
      new File(tmpFolder, "targetDir.zip"),
      new File(Paths.get(tmpFolder).toString))

    assert(new File(tmpFolder, "targetDir.zip").exists())
  }

  "zipFile" should "zip a single file" taggedAs FastTest in {
    ZipArchiveUtil.zipFile(
      new File(Paths.get(tmpFolder, "dir", "fileA").toString),
      new File(tmpFolder, "targetA.zip"))
    assert(new File(tmpFolder, "targetA.zip").exists())
  }

  "zipDir" should "zip a directory" taggedAs FastTest in {
    ZipArchiveUtil.zipDir(
      new File(Paths.get(tmpFolder, "dir").toString),
      new File(tmpFolder, "targetDir.zip"))
    assert(new File(tmpFolder, "targetDir.zip").exists())
  }

  "zip" should "zip a single file with String input" taggedAs FastTest in {
    ZipArchiveUtil.zip(
      Paths.get(tmpFolder, "dir", "fileA").toString,
      Paths.get(tmpFolder, "targetA.zip").toString)
    assert(new File(tmpFolder, "targetA.zip").exists())
  }

  "zip" should "zip a dir with String input" taggedAs FastTest in {
    ZipArchiveUtil.zip(
      Paths.get(tmpFolder, "dir").toString,
      Paths.get(tmpFolder, "targetDir.zip").toString)
    assert(new File(tmpFolder, "targetDir.zip").exists())
  }

  "zip" should "throw exception if the folder not exist since we are not responsible to create folders" taggedAs FastTest in {
    val isIOExceptinoCaught =
      try {
        ZipArchiveUtil.zip(
          Paths.get(tmpFolder, "dir").toString,
          Paths.get(tmpFolder, "otherdir/targetDir.zip").toString)
        false
      } catch {
        case e: java.io.IOException => true
        case _: Throwable => false
      }

    assert(isIOExceptinoCaught)
  }
}
