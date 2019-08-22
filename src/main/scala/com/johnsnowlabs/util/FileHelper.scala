package com.johnsnowlabs.util

import java.io.{File, IOException}
import java.nio.charset.Charset
import java.nio.file.{Files, Paths}
import java.security.MessageDigest
import java.text.DecimalFormat

import org.apache.commons.io.FileUtils
object FileHelper {
  def writeLines(file: String, lines: Seq[String], encoding: String = "UTF-8"): Unit = {
    val writer = Files.newBufferedWriter(Paths.get(file), Charset.forName(encoding))
    try {
      var cnt = 0
      for (line <- lines) {
        writer.write(line)
        if (cnt > 0)
          writer.write(System.lineSeparator())
        cnt += 1
      }
    }
    catch {
      case ex: IOException =>
        ex.printStackTrace()
    }
    finally if (writer != null) writer.close()
  }

  def delete(file: String, throwOnError: Boolean = false): Unit = {
    val f = new File(file)
    if (f.exists()) {
      try {
        if (f.isDirectory)
          FileUtils.deleteDirectory(f)
        else
          FileUtils.deleteQuietly(f)
      }
      catch {
        case e: Exception =>
          if (throwOnError)
            throw e
          else
            FileUtils.forceDeleteOnExit(f)
      }
    }

  }

  def generateChecksum(path: String): String = {
    val arr = Files readAllBytes (Paths get path)
    val checksum = MessageDigest.getInstance("MD5") digest arr
    checksum.map("%02X" format _).mkString
  }

  def getHumanReadableFileSize(size: Long): String = {
    if (size <= 0) return "0"
    val units = Array[String]("B", "KB", "MB", "GB", "TB", "PB", "EB")
    val digitGroups = (Math.log10(size) / Math.log10(1024)).toInt
    new DecimalFormat("#,##0.#").format(size / Math.pow(1024, digitGroups)) + " " + units(digitGroups)
  }
}
