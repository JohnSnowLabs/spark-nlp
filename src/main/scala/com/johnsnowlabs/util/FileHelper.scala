package com.johnsnowlabs.util

import java.io.IOException
import java.nio.charset.Charset
import java.nio.file.{Files, Paths}

object FileHelper {
  def writeLines(file: String, lines: Seq[String], encoding: String = "UTF-8"): Unit = {
    val writer = Files.newBufferedWriter(Paths.get(file), Charset.forName("UTF-8"))
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

}
