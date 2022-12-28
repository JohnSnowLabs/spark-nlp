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

package com.johnsnowlabs.ml.tensorflow.io

import java.io.{BufferedOutputStream, FileInputStream, FileOutputStream, IOException}
import java.nio.file.{Files, Path}
import scala.collection.mutable.ArrayBuffer

private[johnsnowlabs] object ChunkBytes {

  /** readFileInByteChunks will read a file by chuning the size of BufferSize and return array of
    * arrays of Byte
    *
    * @param inputPath
    *   the path to an input file
    * @param BufferSize
    *   the size of bytes in each chunk
    * @return
    *   Array of Arrays of Byte
    */
  def readFileInByteChunks(inputPath: Path, BufferSize: Int = 1024 * 1024): Array[Array[Byte]] = {

    val fis = new FileInputStream(inputPath.toString)
    val sbc = Files.newByteChannel(inputPath).size()
    val MAX_FILE_SIZE = Integer.MAX_VALUE - 8

    if (sbc < MAX_FILE_SIZE) {
      Array(Files.readAllBytes(inputPath))
    } else {
      val varBytesBuffer = new ArrayBuffer[Array[Byte]]()
      var read = 0
      do {
        var chunkBuffer = new Array[Byte](BufferSize)
        read = fis.read(chunkBuffer, 0, BufferSize)
        varBytesBuffer append chunkBuffer
        chunkBuffer = null
      } while (read > -1)

      fis.close()
      varBytesBuffer.toArray
    }

  }

  /** writeByteChunksInFile will write chunks of bytes into a file
    *
    * @param outputPath
    *   the path to an output file
    * @param chunkBytes
    *   the array containing chunks of bytes
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
    } catch {
      case e: IOException =>
        e.printStackTrace()
    }
  }

}
