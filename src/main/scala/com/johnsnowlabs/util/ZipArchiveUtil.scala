/*
 * Copyright 2017-2021 John Snow Labs
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

import scala.collection.JavaConverters._
import java.util.zip.{ZipEntry, ZipFile, ZipOutputStream}
import java.io.{BufferedInputStream, File, FileInputStream, FileOutputStream, IOException}

object ZipArchiveUtil {

  private def listFiles(file: File, outputFilename: String): List[String] = {
    file match {
      case file if file.isFile =>
        if (file.getName != outputFilename)
          List(file.getAbsoluteFile.toString)
        else
          List()
      case file if file.isDirectory =>
        val fList = file.list
        // Add all files in current dir to list and recur on subdirs
        fList.foldLeft(List[String]())((pList: List[String], path: String) =>
          pList ++ listFiles(new File(file, path), outputFilename))
      case _ => throw new IOException("Bad path. No file or directory found.")
    }
  }

  private def addFileToZipEntry(filename: String, parentPath: String,
                                filePathsCount: Int): ZipEntry = {
    if (filePathsCount <= 1)
      new ZipEntry(new File(filename).getName)
    else {
      // use relative path to avoid adding absolute path directories
      val relative = new File(parentPath).toURI.
        relativize(new File(filename).toURI).getPath
      new ZipEntry(relative)
    }
  }

  private def createZip(filePaths: List[String], outputFilename: String, parentPath: String): Unit = {

    val Buffer = 2 * 1024
    val data = new Array[Byte](Buffer)
    try {
      val zipFileOS = new FileOutputStream(outputFilename)
      val zip = new ZipOutputStream(zipFileOS)
      zip.setLevel(0)
      filePaths.foreach((name: String) => {
        val zipEntry = addFileToZipEntry(name, parentPath, filePaths.size)
        //add zip entry to output stream
        zip.putNextEntry(new ZipEntry(zipEntry))
        val in = new BufferedInputStream(new FileInputStream(name), Buffer)
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

  def zip(fileName: String, outputFileName: String): Unit = {
    val file = new File(fileName)
    val filePaths = listFiles(file, outputFileName)
    createZip(filePaths, outputFileName, fileName)
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
    }
    else {
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
      FileUtils.copyInputStreamToFile(zip.getInputStream(entry),
        new File(destDir, entryPath))
    }

    destDir.getPath
  }
}