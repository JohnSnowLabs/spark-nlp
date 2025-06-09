package com.johnsnowlabs.ml.ai.util.transform

import java.awt.image.BufferedImage
import java.awt.{Color, Graphics2D}

object ChannelDimension extends Enumeration {
  type ChannelDimension = Value
  val FIRST, LAST = Value
}

object E5VUtils {
  import ChannelDimension._

  def selectBestResolution(
      originalSize: (Int, Int),
      possibleResolutions: List[(Int, Int)]): (Int, Int) = {
    val (originalHeight, originalWidth) = originalSize
    var bestFit: (Int, Int) = possibleResolutions.head
    var maxEffectiveResolution = 0
    var minWastedResolution = Double.PositiveInfinity

    for ((height, width) <- possibleResolutions) {
      val scale = math.min(width.toDouble / originalWidth, height.toDouble / originalHeight)
      val downscaledWidth = (originalWidth * scale).toInt
      val downscaledHeight = (originalHeight * scale).toInt
      val effectiveResolution =
        math.min(downscaledWidth * downscaledHeight, originalWidth * originalHeight)
      val wastedResolution = (width * height) - effectiveResolution

      if (effectiveResolution > maxEffectiveResolution ||
        (effectiveResolution == maxEffectiveResolution && wastedResolution < minWastedResolution)) {
        maxEffectiveResolution = effectiveResolution
        minWastedResolution = wastedResolution
        bestFit = (height, width)
      }
    }
    bestFit
  }

  def imageSizeToNumPatches(
      imageSize: (Int, Int),
      gridPinpoints: List[(Int, Int)],
      patchSize: Int): Int = {
    val (height, width) = selectBestResolution(imageSize, gridPinpoints)
    val numPatches = (0 until height by patchSize).size * (0 until width by patchSize).size
    // add the base patch
    numPatches + 1
  }

  def getAnyResImageGridShape(
      imageSize: (Int, Int),
      gridPinpoints: List[(Int, Int)],
      patchSize: Int): (Int, Int) = {
    val (height, width) = selectBestResolution(imageSize, gridPinpoints)
    (height / patchSize, width / patchSize)
  }

  def getImageSize(image: BufferedImage): (Int, Int) = {
    (image.getHeight, image.getWidth)
  }

  def expandToSquare(image: BufferedImage, backgroundColor: Color): BufferedImage = {
    val width = image.getWidth
    val height = image.getHeight
    if (width == height) {
      image
    } else if (width > height) {
      val result = new BufferedImage(width, width, image.getType)
      val g = result.createGraphics()
      g.setColor(backgroundColor)
      g.fillRect(0, 0, width, width)
      g.drawImage(image, 0, (width - height) / 2, null)
      g.dispose()
      result
    } else {
      val result = new BufferedImage(height, height, image.getType)
      val g = result.createGraphics()
      g.setColor(backgroundColor)
      g.fillRect(0, 0, height, height)
      g.drawImage(image, (height - width) / 2, 0, null)
      g.dispose()
      result
    }
  }

  def divideToPatches(image: BufferedImage, patchSize: Int): List[BufferedImage] = {
    val width = image.getWidth
    val height = image.getHeight
    val patches = for {
      i <- 0 until height by patchSize
      j <- 0 until width by patchSize
    } yield {
      val w = math.min(patchSize, width - j)
      val h = math.min(patchSize, height - i)
      image.getSubimage(j, i, w, h)
    }
    patches.toList
  }

  def getPatchOutputSize(image: BufferedImage, targetResolution: (Int, Int)): (Int, Int) = {
    val (originalHeight, originalWidth) = getImageSize(image)
    val (targetHeight, targetWidth) = targetResolution

    val scaleW = targetWidth.toDouble / originalWidth
    val scaleH = targetHeight.toDouble / originalHeight

    if (scaleW < scaleH) {
      val newWidth = targetWidth
      val newHeight = math.min(math.ceil(originalHeight * scaleW).toInt, targetHeight)
      (newHeight, newWidth)
    } else {
      val newHeight = targetHeight
      val newWidth = math.min(math.ceil(originalWidth * scaleH).toInt, targetWidth)
      (newHeight, newWidth)
    }
  }

  def padImage(image: BufferedImage, targetResolution: (Int, Int)): BufferedImage = {
    val (targetHeight, targetWidth) = targetResolution
    val (originalHeight, originalWidth) = getImageSize(image)
    val (newHeight, newWidth) = getPatchOutputSize(image, targetResolution)
    val result = new BufferedImage(targetWidth, targetHeight, image.getType)
    val g = result.createGraphics()
    g.setColor(Color.BLACK)
    g.fillRect(0, 0, newWidth, newHeight)
    g.drawImage(
      image,
      (targetWidth - originalWidth) / 2,
      (targetHeight - originalHeight) / 2,
      null)
    g.dispose()
    result
  }
}
