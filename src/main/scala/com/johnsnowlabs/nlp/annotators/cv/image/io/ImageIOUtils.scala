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

package com.johnsnowlabs.nlp.annotators.cv.image.io

import java.awt.Point
import java.awt.image.{BufferedImage, DataBufferByte, Raster}
import java.io.File
import javax.imageio.ImageIO

private[johnsnowlabs] object ImageIOUtils {

  def loadImage(path: String): BufferedImage = {
    ImageIO.read(new File(path))
  }

  def convertChannelsToType(channels: Int): Int = channels match {
    case 1 => BufferedImage.TYPE_BYTE_GRAY
    case 3 => BufferedImage.TYPE_3BYTE_BGR
    case 4 => BufferedImage.TYPE_4BYTE_ABGR
    case c =>
      throw new UnsupportedOperationException(
        "Image resize: number of output  " +
          s"channels must be 1, 3, or 4, got $c.")
  }

  def byteToBufferedImage(bytes: Array[Byte], w: Int, h: Int, nChannels: Int): BufferedImage = {
    val img = new BufferedImage(w, h, convertChannelsToType(nChannels))
    img.setData(
      Raster
        .createRaster(img.getSampleModel, new DataBufferByte(bytes, bytes.length), new Point()))
    img
  }

}
