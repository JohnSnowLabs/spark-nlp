/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.Path

import java.io.{File, FileWriter, PrintWriter}
import scala.language.existentials


object OutputHelper {

  private lazy val fileSystem = ConfigHelper.getFileSystem

  private def logsFolder: String = ConfigLoader.getConfigStringValue(ConfigHelper.annotatorLogFolder)

  lazy private val isDBFS = fileSystem.getScheme.equals("dbfs")

  def writeAppend(uuid: String, content: String, outputLogsPath: String): Unit = {
    val targetFolder = if (outputLogsPath.isEmpty) logsFolder else outputLogsPath

    if (isDBFS) {
      if (!new File(targetFolder).exists()) new File(targetFolder).mkdirs()
    } else {
      if (!fileSystem.exists(new Path(targetFolder))) fileSystem.mkdirs(new Path(targetFolder))
    }

    val targetPath = new Path(targetFolder, uuid + ".log")

    if (fileSystem.getScheme.equals("file") || fileSystem.getScheme.equals("dbfs")) {
      val fo = new File(targetPath.toUri.getRawPath)
      val writer = new FileWriter(fo, true)
      writer.append(content + System.lineSeparator())
      writer.close()
    } else {
      fileSystem.createNewFile(targetPath)
      val fo = fileSystem.append(targetPath)
      val writer = new PrintWriter(fo, true)
      writer.append(content + System.lineSeparator())
      writer.close()
      fo.close()

    }
  }

}
