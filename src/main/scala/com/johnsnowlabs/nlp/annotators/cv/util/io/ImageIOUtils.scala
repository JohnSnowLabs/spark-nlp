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

package com.johnsnowlabs.nlp.annotators.cv.util.io

import com.johnsnowlabs.nlp.ImageFields
import com.johnsnowlabs.nlp.util.io.ResourceHelper

import java.awt.color.ColorSpace
import java.awt.image.{BufferedImage, DataBufferByte, Raster}
import java.awt.{Color, Point}
import java.io.{File, InputStream}
import javax.imageio.ImageIO
import scala.util.{Failure, Success, Try}

private[johnsnowlabs] object ImageIOUtils {

  /** (Scala-specific) OpenCV type mapping supported */
  val ocvTypes: Map[String, Int] =
    Map("CV_8U" -> 0, "CV_8UC1" -> 0, "CV_8UC3" -> 16, "CV_8UC4" -> 24)

  def loadImage(file: File): BufferedImage = {
    ImageIO.read(file)
  }

  def loadImage(file: InputStream): BufferedImage = {
    ImageIO.read(file)
  }

  def loadImage(path: String): BufferedImage = {
    val filePath = getFileFromPath(path)
    ImageIO.read(filePath)
  }

  def loadImageFromAnySource(path: String): BufferedImage = {

    val prefix = if (path.indexOf(":") == -1) "" else path.substring(path.indexOf(":"))

    prefix match {
      case "dbfs" =>
        loadImage(path.replace("dbfs:", "/dbfs/"))
      case "hdfs" =>
        val sourceStream = ResourceHelper.SourceStream(path)
        ImageIO.read(sourceStream.pipe.head)
      case _ =>
        loadImage(path)
    }

  }

  def loadImages(imagesPath: String): Array[File] = {
    loadImagesFromDirectory(imagesPath) match {
      case Success(files) => files
      case Failure(_) =>
        val singleImagePath = getFileFromPath(imagesPath)
        Array(singleImagePath)
    }
  }

  private def loadImagesFromDirectory(path: String): Try[Array[File]] = {
    Try {
      ResourceHelper.listLocalFiles(path).toArray
    }
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

  def bufferedImageToByte(img: BufferedImage): Array[Byte] = {

    if (img == null) {
      Array.empty[Byte]
    } else {

      val is_gray = img.getColorModel.getColorSpace.getType == ColorSpace.TYPE_GRAY

      val height = img.getHeight
      val width = img.getWidth
      val (nChannels, _) = getChannelsAndMode(img)

      assert(height * width * nChannels < 1e9, "image is too large")
      val decoded = Array.ofDim[Byte](height * width * nChannels)

      // grayscale images in Java require special handling to get the correct intensity
      if (is_gray) {
        var offset = 0
        val raster = img.getRaster
        for (h <- 0 until height) {
          for (w <- 0 until width) {
            decoded(offset) = raster.getSample(w, h, 0).toByte
            offset += 1
          }
        }
      } else {
        var offset = 0
        for (h <- 0 until height) {
          for (w <- 0 until width) {
            val color = new Color(img.getRGB(w, h))

            decoded(offset) = color.getBlue.toByte
            decoded(offset + 1) = color.getGreen.toByte
            decoded(offset + 2) = color.getRed.toByte
            if (nChannels == 4) {
              decoded(offset + 3) = color.getAlpha.toByte
            }
            offset += nChannels
          }
        }
      }
      decoded
    }
  }

  private def getChannelsAndMode(bufferedImage: BufferedImage): (Int, Int) = {
    val is_gray = bufferedImage.getColorModel.getColorSpace.getType == ColorSpace.TYPE_GRAY
    val has_alpha = bufferedImage.getColorModel.hasAlpha

    val (numberOfChannels, mode) =
      if (is_gray) (1, ocvTypes.getOrElse("CV_8UC1", -1))
      else if (has_alpha) (4, ocvTypes.getOrElse("CV_8UC4", -1))
      else (3, ocvTypes.getOrElse("CV_8UC3", -1))

    (numberOfChannels, mode)
  }

  def imagePathToImageFields(imagePath: String): ImageFields = {
    val bufferedImage = loadImageFromAnySource(imagePath)
    bufferedImageToImageFields(bufferedImage, imagePath)
  }

  def imageFileToImageFields(file: File): ImageFields = {
    val bufferedImage = loadImage(file)
    bufferedImageToImageFields(bufferedImage, file.getPath)
  }

  def bufferedImageToImageFields(bufferedImage: BufferedImage, origin: String): ImageFields = {
    val (nChannels, mode) = getChannelsAndMode(bufferedImage)
    val data = bufferedImageToByte(bufferedImage)

    ImageFields(origin, bufferedImage.getHeight, bufferedImage.getWidth, nChannels, mode, data)
  }

  def getFileFromPath(path: String): File = {
    val filePath = new File(path)
    if (!filePath.exists()) {
      throw new IllegalArgumentException(f"Wrong image file path $path")
    }

    filePath
  }

}
