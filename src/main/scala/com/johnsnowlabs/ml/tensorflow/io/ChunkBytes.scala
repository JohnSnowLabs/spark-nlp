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

package com.johnsnowlabs.ml.tensorflow.io

import java.io.{BufferedOutputStream, FileInputStream, FileOutputStream, IOException}
import java.nio.file.{Files, Path}

object ChunkBytes {

  /**
   * readFileInByteChunks will read a file by chuning the size of BufferSize and return array of arrays of Byte
   *
   * @param inputPath  the path to an input file
   * @param BufferSize the size of bytes in each chunk
   * @return Array of Arrays of Byte
   */
  def readFileInByteChunks(inputPath: Path, BufferSize: Int = 1024 * 1024): Array[Array[Byte]] = {

    val fis = new FileInputStream(inputPath.toString)
    val sbc = Files.newByteChannel(inputPath).size()
    val MAX_FILE_SIZE = Integer.MAX_VALUE - 8

    if (sbc < MAX_FILE_SIZE) {
      Array(Files.readAllBytes(inputPath))
    } else {
      val chunksCount = (sbc / BufferSize).toInt + 1

      val varBytes = Array.ofDim[Array[Byte]](chunksCount)
      val buffer = Array.ofDim[Byte](BufferSize)

      var read = fis.read(buffer, 0, BufferSize)
      var index = 0

      while (read > -1) {
        varBytes(index) = buffer.clone()
        read = fis.read(buffer, 0, BufferSize)
        index += 1
      }
      fis.close()
      varBytes
    }

  }

  /**
   * writeByteChunksInFile will write chunks of bytes into a file
   *
   * @param outputPath the path to an output file
   * @param chunkBytes the array containing chunks of bytes
   */
  def writeByteChunksInFile(outputPath: Path, chunkBytes: Array[Array[Byte]]): Unit = {

    try {
      val out = new BufferedOutputStream(new FileOutputStream(outputPath.toString))
      var count = 0

      while (count < chunkBytes.length) {
        val bytes = chunkBytes(count)
        out.write(bytes, 0, bytes.length)
        count += 1
      }
      out.close()
    }
    catch {
      case e: IOException =>
        e.printStackTrace()
    }
  }

}
