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

import java.awt.Color
import java.awt.geom.AffineTransform
import java.awt.image.{AffineTransformOp, BufferedImage}
import scala.collection.mutable.ArrayBuffer

private[johnsnowlabs] object ImageResizeUtils {

  /** Resized a BufferedImage with specified width, height and filter.
    *
    * @param width
    *   Target width of the image
    * @param height
    *   Target height of the image
    * @param resample
    *   Transformation/Filter to apply, either `AffineTransformOp.TYPE_NEAREST_NEIGHBOR`(1),
    *   `AffineTransformOp.TYPE_BILINEAR`(2), `AffineTransformOp.TYPE_BICUBIC`(3)
    * @param image
    *   Image to resize
    * @return
    *   Resized BufferedImage
    */
  def resizeBufferedImage(width: Int, height: Int, resample: Int)(
      image: BufferedImage): BufferedImage = {

    val scaleX = width / image.getWidth.toDouble
    val scaleY = height / image.getHeight.toDouble

    val transform = AffineTransform.getScaleInstance(scaleX, scaleY)
    val transformOp = new AffineTransformOp(transform, resample)
    val result = transformOp.filter(image, null) // Creates new BufferedImage

    result
  }

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
  def normalizeAndConvertBufferedImage(
      img: BufferedImage,
      mean: Array[Double],
      std: Array[Double],
      doNormalize: Boolean,
      doRescale: Boolean,
      rescaleFactor: Double): Array[Array[Array[Float]]] = {

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

        // Retrieving the R G B values, rescaling and normalizing them
        val rescaledRed = if (doRescale) color.getRed * rescaleFactor else color.getRed
        val rescaledGreen = if (doRescale) color.getGreen * rescaleFactor else color.getGreen
        val rescaledBlue = if (doRescale) color.getBlue * rescaleFactor else color.getBlue

        val (red, green, blue) = if (doNormalize) {
          val normR = (rescaledRed - mean.head) / std.head
          val normG = (rescaledGreen - mean(1)) / std(1)
          val normB = (rescaledBlue - mean(2)) / std(2)
          (normR, normG, normB)
        } else (rescaledRed, rescaledGreen, rescaledBlue)

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

  /** Resizes and crops an image, intended for smaller images.
    *
    * The image is resized based on a percentage. The smaller edge will be resized to
    * `requestedSize / cropPct` and then cropped to `(requestedSize, requestedSize)`.
    */
  def resizeAndCenterCropImage(
      img: BufferedImage,
      requestedSize: Int,
      resample: Int,
      cropPct: Double): BufferedImage = {

    val width = img.getWidth()
    val height = img.getHeight()

    val (shortEdge, longEdge) = if (width <= height) (width, height) else (height, width)

    val sizeForCrop = requestedSize / cropPct
    val newShortEdge = sizeForCrop.toInt
    val newLongEdge = (sizeForCrop * (longEdge / shortEdge.toFloat)).toInt

    val (resizeWidth, resizeHeight) =
      if (width <= height) (newShortEdge, newLongEdge) else (newLongEdge, newShortEdge)

    // Resize the Image with the new calculated size
    val resizedImage =
      resizeBufferedImage(resizeWidth, resizeHeight, resample)(img)

    // Crop at the center of the image
    val cropLeft = (resizeWidth - requestedSize).abs / 2
    val cropTop = (resizeHeight - requestedSize).abs / 2

    cropBufferedImage(resizedImage, cropLeft, cropTop, requestedSize, requestedSize)
  }
}
