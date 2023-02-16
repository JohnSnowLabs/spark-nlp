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

package com.johnsnowlabs.nlp.annotators.cv.util.transform

import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils

import java.awt.geom.AffineTransform
import java.awt.image.{AffineTransformOp, BufferedImage}
import java.awt.{Color, Image}
import scala.collection.mutable.ArrayBuffer

private[johnsnowlabs] object ImageResizeUtils {

  def resizeBufferedImage(width: Int, height: Int, channels: Option[Int])(
      image: BufferedImage): BufferedImage = {
    val imgType = channels.map(ImageIOUtils.convertChannelsToType).getOrElse(image.getType)

    if (image.getWidth == width &&
      image.getHeight == height &&
      image.getType == imgType) {
      return image
    }

    // SCALE_AREA_AVERAGING performs slower than SCALE_DEFAULT
    val resizedImage = image.getScaledInstance(width, height, Image.SCALE_DEFAULT)
    val bufferedImage = new BufferedImage(width, height, imgType)
    val graphic = bufferedImage.createGraphics()
    // scalastyle:ignore null
    graphic.drawImage(resizedImage, 0, 0, null)
    graphic.dispose()
    bufferedImage
  }

  // TODO implement doNormalize = false to only return Array[Array[Array[Float]]] without normalizing
  /** @param img
    *   The image in BufferedImage
    * @param mean
    *   Mean to subtract
    * @param std
    *   Standard deviation to normalize
    * @param rescaleFactor
    *   Factor to rescale the image values by
    * @return
    */
  def normalizeBufferedImage(
      img: BufferedImage,
      mean: Array[Double],
      std: Array[Double],
      rescaleFactor: Double = 1 / 255.0d): Array[Array[Array[Float]]] = {
    val data =
      Array(ArrayBuffer[Array[Float]](), ArrayBuffer[Array[Float]](), ArrayBuffer[Array[Float]]())
    for (y <- 0 until img.getHeight) {
      val RedList = ArrayBuffer[Float]()
      val GreenList = ArrayBuffer[Float]()
      val BlueList = ArrayBuffer[Float]()
      for (x <- 0 until img.getWidth) {
        // Retrieving contents of a pixel
        val pixel = img.getRGB(x, y)
        // Creating a Color object from pixel value
        val color = new Color(pixel, true)

        // Retrieving the R G B values and Normalizing them
        val red = ((color.getRed * rescaleFactor) - mean.head) / std.head
        val green = ((color.getGreen * rescaleFactor) - mean(1)) / std(1)
        val blue = ((color.getBlue * rescaleFactor) - mean(2)) / std(2)
        RedList += red.toFloat
        GreenList += green.toFloat
        BlueList += blue.toFloat
      }
      data.head += RedList.toArray
      data(1) += GreenList.toArray
      data(2) += BlueList.toArray
    }
    data.map(_.toArray)
  }

  def resampleBufferedImage(image: BufferedImage): BufferedImage = {
    val w = image.getWidth
    val h = image.getHeight
    var scaledImage = new BufferedImage(w * 2, h * 2, BufferedImage.TYPE_INT_ARGB)
    val at = AffineTransform.getScaleInstance(2.0, 2.0)
    val ato =
      new AffineTransformOp(
        at,
        AffineTransformOp.TYPE_NEAREST_NEIGHBOR
      ) // Currently we have three types : TYPE_BICUBIC TYPE_BILINEAR and TYPE_NEAREST_NEIGHBOR
    scaledImage = ato.filter(image, scaledImage)
    scaledImage
  }

  /** Crops an image to the specified region
    *
    * @param bufferedImage
    *   the image that will be crop
    * @param x
    *   the upper left x coordinate that this region will start
    * @param y
    *   the upper left y coordinate that this region will start
    * @param width
    *   the width of the region that will be crop
    * @param height
    *   the height of the region that will be crop
    * @return
    *   the image that was cropped.
    */
  def cropBufferedImage(
      bufferedImage: BufferedImage,
      x: Int,
      y: Int,
      width: Int,
      height: Int): BufferedImage = {
    bufferedImage.getSubimage(x, y, width, height)
  }
}
