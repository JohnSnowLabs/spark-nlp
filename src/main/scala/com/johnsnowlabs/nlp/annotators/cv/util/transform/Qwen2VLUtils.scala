package com.johnsnowlabs.nlp.annotators.cv.util.transform
import java.awt.image.BufferedImage

private[johnsnowlabs] object Qwen2VLUtils {

  val IMAGE_FACTOR: Int = 28
  val MIN_PIXELS: Int = 4 * 28 * 28
  val MAX_PIXELS: Int = 16384 * 28 * 28
  val MAX_RATIO: Int = 200

  def roundByFactor(number: Int, factor: Int): Int =
    Math.round(number.toDouble / factor).toInt * factor

  def ceilByFactor(number: Int, factor: Int): Int =
    Math.ceil(number.toDouble / factor).toInt * factor

  def floorByFactor(number: Int, factor: Int): Int =
    Math.floor(number.toDouble / factor).toInt * factor

  def smartResize(
      height: Int,
      width: Int,
      factor: Int = IMAGE_FACTOR,
      minPixels: Int = MIN_PIXELS,
      maxPixels: Int = MAX_PIXELS): (Int, Int) = {
    if (Math.max(height, width).toDouble / Math.min(height, width) > MAX_RATIO) {
      throw new IllegalArgumentException(s"absolute aspect ratio must be smaller than $MAX_RATIO")
    }

    var hBar = Math.max(factor, roundByFactor(height, factor))
    var wBar = Math.max(factor, roundByFactor(width, factor))

    if (hBar * wBar > maxPixels) {
      val beta = Math.sqrt(height.toDouble * width / maxPixels)
      hBar = floorByFactor((height / beta).toInt, factor)
      wBar = floorByFactor((width / beta).toInt, factor)
    } else if (hBar * wBar < minPixels) {
      val beta = Math.sqrt(minPixels.toDouble / (height * width))
      hBar = ceilByFactor((height * beta).toInt, factor)
      wBar = ceilByFactor((width * beta).toInt, factor)
    }

    (hBar, wBar)
  }

  def imageBufferToArray(imgCrop: BufferedImage): Array[Array[Array[Int]]] = {
    val height = imgCrop.getHeight
    val width = imgCrop.getWidth

    // Create a 3D array for RGB channels
    val channels = 3
    val cropArray = Array.ofDim[Int](channels, height, width)

    for (y <- 0 until height; x <- 0 until width) {
      val color = new java.awt.Color(imgCrop.getRGB(x, y))
      cropArray(0)(y)(x) = color.getRed // Red channel
      cropArray(1)(y)(x) = color.getGreen // Green channel
      cropArray(2)(y)(x) = color.getBlue // Blue channel
    }

    cropArray
  }
}
