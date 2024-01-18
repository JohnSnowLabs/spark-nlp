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

import org.apache.commons.io.FileUtils

import java.io._
import java.util.zip.{ZipEntry, ZipFile, ZipOutputStream}
import scala.collection.JavaConverters._

object ZipArchiveUtil {
  // Recursively lists all files in a given directory, returning a list of their absolute paths
  private[util] def listFilesRecursive(file: File): List[File] = {
    file match {
      case file if file.isFile => List(new File(file.getAbsoluteFile.toString))
      case file if file.isDirectory =>
        val fList = file.list
        // Add all files in current dir to list and recur on subdirs
        fList.foldLeft(List[File]())((pList: List[File], path: String) =>
          pList ++ listFilesRecursive(new File(file, path)))
      case _ => throw new IOException("Bad path. No file or directory found.")
    }
  }

  private[util] def addFileToZipEntry(
      filename: File,
      parentPath: File,
      useRelativePath: Boolean = false): ZipEntry = {
    if (!useRelativePath) // use absolute path
      new ZipEntry(filename.getName)
    else { // use relative path
      val relative = parentPath.toURI.relativize(filename.toURI).getPath
      new ZipEntry(relative)
    }
  }

  private[util] def createZip(
      filePaths: List[File],
      outputFilePath: File,
      parentPath: File): Unit = {

    val Buffer = 2 * 1024
    val data = new Array[Byte](Buffer)
    try {
      val zipFileOS = new FileOutputStream(outputFilePath)
      val zip = new ZipOutputStream(zipFileOS)
      zip.setLevel(0)
      filePaths.foreach((file: File) => {
        val zipEntry = addFileToZipEntry(file, parentPath, filePaths.size > 1)
        // add zip entry to output stream
        zip.putNextEntry(new ZipEntry(zipEntry))
        val in = new BufferedInputStream(new FileInputStream(file), Buffer)
        var b = in.read(data, 0, Buffer)
        while (b != -1) {
          zip.write(data, 0, b)
          b = in.read(data, 0, Buffer)
        }
        in.close()
      })
      zip.closeEntry()
      zip.close()
      zipFileOS.close()

    } catch {
      case e: IOException =>
        e.printStackTrace()
    }
  }

  private[util] def zipFile(soureFile: File, outputFilePath: File): Unit = {
    createZip(List(soureFile.getAbsoluteFile), outputFilePath, null)
  }

  private[util] def zipDir(sourceDir: File, outputFilePath: File): Unit = {
    val filePaths = listFilesRecursive(sourceDir)
    createZip(filePaths, outputFilePath, sourceDir)
  }

  def zip(sourcePath: String, outputFilePath: String): Unit = {
    val sourceFile = new File(sourcePath)
    val outputFile = new File(outputFilePath)
    if (sourceFile.equals(outputFile))
      throw new IllegalArgumentException("source path cannot be identical to target path")

    if (!outputFile.getParentFile().exists)
      throw new IOException("the parent directory of output file doesn't exist")

    if (!sourceFile.exists())
      throw new IOException("zip source path must exsit")

    if (outputFile.exists())
      throw new IOException("zip target file exsits")

    if (sourceFile.isDirectory())
      zipDir(sourceFile, outputFile)
    else if (sourceFile.isFile())
      zipFile(sourceFile, outputFile)
    else
      throw new IllegalArgumentException("only folder and file input are valid")
  }

  def unzip(file: File, destDirPath: Option[String] = None): String = {
    val fileName = file.getName

    val basename = if (fileName.indexOf('.') >= 0) {
      fileName.substring(0, fileName.lastIndexOf("."))
    } else {
      fileName + "_unzipped"
    }

    val destDir = if (destDirPath.isEmpty) {
      new File(file.getParentFile, basename)
    } else {
      new File(destDirPath.get)
    }

    destDir.mkdirs()

    val zip = new ZipFile(file)
    zip.entries.asScala foreach { entry =>
      val entryName = entry.getName
      val entryPath = {
        if (entryName.startsWith(basename))
          entryName.substring(basename.length)
        else
          entryName
      }

      // create output directory if it doesn't exist already
      val toDrop = if (entry.isDirectory) 0 else 1
      val splitPath = entryName.split(File.separator.replace("\\", "/")).dropRight(toDrop)

      val dirBuilder = new StringBuilder(destDir.getPath)
      for (part <- splitPath) {
        dirBuilder.append(File.separator)
        dirBuilder.append(part)
        val path = dirBuilder.toString

        if (!new File(path).exists) {
          new File(path).mkdir
        }
      }

      // write file to dest
      FileUtils.copyInputStreamToFile(zip.getInputStream(entry), new File(destDir, entryPath))
    }

    destDir.getPath
  }
}
