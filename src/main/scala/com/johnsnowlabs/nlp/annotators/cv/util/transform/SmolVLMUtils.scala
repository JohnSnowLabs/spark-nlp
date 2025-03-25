package com.johnsnowlabs.nlp.annotators.cv.util.transform
import java.awt.image.BufferedImage
import java.awt.{Color, Graphics2D}
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer

import ImageResizeUtils.resizeBufferedImage

private[johnsnowlabs] object SmolVLMUtils {

  def resizeImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    val resizedImage = new BufferedImage(width, height, image.getType)
    val g = resizedImage.createGraphics()
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    resizedImage
  }

  def _crop(image: BufferedImage, x1: Int, y1: Int, x2: Int, y2: Int): BufferedImage = {
    val width = x2 - x1
    val height = y2 - y1
    val croppedImage = new BufferedImage(width, height, image.getType)
    val g = croppedImage.createGraphics()
    g.drawImage(image, 0, 0, width, height, x1, y1, x2, y2, null)
    g.dispose()
    croppedImage
  }

  case class SplitImageResult(frames: Seq[BufferedImage], numSplitsH: Int, numSplitsW: Int)

  def splitImage(image: BufferedImage, maxImageSize: Int, resample: Int = 2): SplitImageResult = {
    val height = image.getHeight
    val width = image.getWidth
    val maxHeight = maxImageSize
    val maxWidth = maxImageSize

    val frames = new ArrayBuffer[BufferedImage]()

    if (height > maxHeight || width > maxWidth) {
      // Calculate the number of splits
      val numSplitsH = math.ceil(height.toDouble / maxHeight).toInt
      val numSplitsW = math.ceil(width.toDouble / maxWidth).toInt

      // Calculate the optimal width and height for the sub-images
      val optimalHeight = math.ceil(height.toDouble / numSplitsH).toInt
      val optimalWidth = math.ceil(width.toDouble / numSplitsW).toInt

      // Iterate through each row and column
      for (r <- 0 until numSplitsH) {
        for (c <- 0 until numSplitsW) {
          // Calculate the starting point of the crop
          val startX = c * optimalWidth
          val startY = r * optimalHeight

          // Calculate the ending point of the crop
          val endX = math.min(startX + optimalWidth, width)
          val endY = math.min(startY + optimalHeight, height)

          // Crop the image
          val croppedImage = _crop(image, startX, startY, endX, endY)
          frames += croppedImage
        }
      }

      // For the global image at the end, we resize it to match the max_image_size
      val resizedImage = resizeBufferedImage(maxWidth, maxHeight, resample)(image)
      frames += resizedImage

      SplitImageResult(frames.toSeq, numSplitsH, numSplitsW)
    } else {
      // If image is smaller than max size, just add the original image
      frames += image
      SplitImageResult(frames.toSeq, 0, 0)
    }
  }

  case class ImageSize(height: Int, width: Int)

  private def resizeOutputSizeRescaleToMaxLen(
      height: Int,
      width: Int,
      minLen: Int = 1,
      maxLen: Option[Int] = None): ImageSize = {
    val effectiveMaxLen = maxLen.getOrElse(math.max(height, width))
    val aspectRatio = width.toDouble / height

    val (newHeight, newWidth) = if (width >= height) {
      val newWidth = effectiveMaxLen
      val newHeight = (newWidth / aspectRatio).toInt
      (if (newHeight % 2 != 0) newHeight + 1 else newHeight, newWidth)
    } else {
      val newHeight = effectiveMaxLen
      val newWidth = (newHeight * aspectRatio).toInt
      (newHeight, if (newWidth % 2 != 0) newWidth + 1 else newWidth)
    }

    // Avoid resizing to a size smaller than minLen
    ImageSize(height = math.max(newHeight, minLen), width = math.max(newWidth, minLen))
  }

  private def resizeOutputSizeScaleBelowUpperBound(
      height: Int,
      width: Int,
      maxLen: Option[Int] = None): ImageSize = {
    val effectiveMaxLen = maxLen.getOrElse(math.max(height, width))
    val aspectRatio = width.toDouble / height

    val (newHeight, newWidth) = if (width >= height && width > effectiveMaxLen) {
      val newWidth = effectiveMaxLen
      val newHeight = (newWidth / aspectRatio).toInt
      (newHeight, newWidth)
    } else if (height > width && height > effectiveMaxLen) {
      val newHeight = effectiveMaxLen
      val newWidth = (newHeight * aspectRatio).toInt
      (newHeight, newWidth)
    } else {
      (height, width)
    }

    // Avoid resizing to a size smaller than 1
    ImageSize(height = math.max(newHeight, 1), width = math.max(newWidth, 1))
  }

  def getResizeOutputImageSize(
      image: BufferedImage,
      resolutionMaxSide: Int,
      maxImageSize: Int): ImageSize = {
    val height = image.getHeight
    val width = image.getWidth

    // Find the output size, when rescaling the longest edge to max_len and preserving the aspect ratio
    val firstResize =
      resizeOutputSizeRescaleToMaxLen(height, width, maxLen = Some(resolutionMaxSide))

    // Find the output size when scaling the image to be below the MAX_IMAGE_SIZE
    resizeOutputSizeScaleBelowUpperBound(
      firstResize.height,
      firstResize.width,
      maxLen = Some(maxImageSize))
  }

  def getMaxHeightWidth(imagesList: Seq[Seq[BufferedImage]]): ImageSize = {
    var maxHeight = Int.MinValue
    var maxWidth = Int.MinValue

    for (images <- imagesList) {
      for (image <- images) {
        val height = image.getHeight
        val width = image.getWidth
        maxHeight = math.max(height, maxHeight)
        maxWidth = math.max(width, maxWidth)
      }
    }

    ImageSize(maxHeight, maxWidth)
  }

  def makePixelMask(image: BufferedImage, outputSize: ImageSize): Array[Array[Int]] = {
    val inputHeight = image.getHeight
    val inputWidth = image.getWidth

    // Create a 2D array filled with zeros
    val mask = Array.ofDim[Int](outputSize.height, outputSize.width)

    // Fill the valid region with ones
    for (y <- 0 until math.min(inputHeight, outputSize.height)) {
      for (x <- 0 until math.min(inputWidth, outputSize.width)) {
        mask(y)(x) = 1
      }
    }

    mask
  }
}
